"""Live interview listener — continuous VAD using sd.InputStream + callback queue.

Audio is captured in a dedicated sounddevice callback thread and queued as
500 ms blocks.  The listener thread drains the queue, gates on RMS, accumulates
speech, and flushes to Whisper when silence is detected.  Recording NEVER pauses
while waiting for the Whisper response because the flush runs on a separate thread.
"""
import queue
import threading
import time
import tempfile
import wave
import os
from asyncio import run_coroutine_threadsafe
from typing import Callable, Optional

try:
    import sounddevice as sd
    import numpy as np
except ImportError:
    sd = None
    np = None


class LiveInterviewListener:
    def __init__(
        self,
        ai,
        loop,
        on_transcript: Optional[Callable[[str], None]] = None,
        on_answer: Optional[Callable[[str], None]] = None,
        on_status: Optional[Callable[[str], None]] = None,
        samplerate: int = 16000,
        channels: int = 1,
        chunk_seconds: float = 0.5,   # kept for API compat; 500 ms blocks used internally
        rms_threshold: float = 0.01,
    ):
        self.ai = ai
        self.loop = loop
        self.on_transcript = on_transcript
        self.on_answer = on_answer
        self.on_status = on_status
        self.samplerate = samplerate
        self.channels = channels
        self.rms_threshold = rms_threshold

        self._thread = None
        self._stop_event = threading.Event()
        self._running = False

    # ── helpers ──────────────────────────────────────────────────────────────

    def _emit(self, msg: str):
        if self.on_status:
            try:
                self.on_status(msg)
            except Exception:
                pass

    # ── public API ───────────────────────────────────────────────────────────

    def start(self):
        if self._running:
            return
        if sd is None or np is None:
            raise RuntimeError("sounddevice or numpy not available")
        self._stop_event.clear()
        self._thread = threading.Thread(target=self._record_loop, daemon=True)
        self._thread.start()
        self._running = True
        self._emit('started')

    def stop(self):
        if not self._running:
            return
        self._stop_event.set()
        if self._thread:
            self._thread.join(timeout=5.0)
        self._running = False
        self._emit('stopped')

    # ── internal ─────────────────────────────────────────────────────────────

    def _write_wav(self, data, sr: int) -> str:
        """Resample to 16 kHz and write a temp WAV."""
        TARGET = 16000
        if sr != TARGET:
            ratio = TARGET / sr
            new_len = max(1, int(len(data) * ratio))
            idx = np.clip(
                (np.arange(new_len) / ratio).astype(np.int64), 0, len(data) - 1
            )
            data = data[idx]
            sr = TARGET
        fd, path = tempfile.mkstemp(suffix=".wav")
        os.close(fd)
        with wave.open(path, "wb") as wf:
            wf.setnchannels(1)
            wf.setsampwidth(2)
            wf.setframerate(sr)
            wf.writeframes(data.tobytes())
        return path

    def _dispatch_flush(self, buf: list, use_sr: int):
        """Run _flush on a throw-away thread so recording continues uninterrupted."""
        threading.Thread(
            target=self._flush, args=(buf, use_sr), daemon=True
        ).start()

    def _flush(self, buf: list, use_sr: int):
        """Concatenate buffer → WAV → Whisper → assistant answer."""
        _JUNK = {
            "", "you", "you.", "thank you", "thank you.", "thanks", "thanks.",
            "bye", "bye.", "bye bye.", ".", "..", "...", "uh", "uh.", "um", "um.",
            "[music]", "[applause]", "subtitles by", "hi", "hi.", "hey", "hey.",
            "thanks for watching", "thanks for watching.", "thank you for watching",
            "thank you for watching.", "thank you so much for watching",
            "thank you so much for watching.", "thanks for watching!",
            "thank you for watching!", "always be happy", "always be happy.",
            "please subscribe", "don't forget to subscribe", "like and subscribe",
            "see you next time", "see you in the next video",
        }
        wav_path = None
        try:
            combined = np.concatenate(buf)
            wav_path = self._write_wav(combined, use_sr)

            fut = run_coroutine_threadsafe(
                self.ai.transcribe(wav_path), self.loop
            )
            try:
                transcript = fut.result(timeout=20)
            except Exception as e:
                self._emit(f"transcribe_error: {e}")
                return

            if not transcript or not transcript.strip():
                return
            if transcript.strip().lower().rstrip(".").strip() in _JUNK:
                self._emit(f"filtered: {transcript!r}")
                return

            try:
                if self.on_transcript:
                    self.on_transcript(transcript)
            except Exception:
                pass

            try:
                reply_fut = run_coroutine_threadsafe(
                    self.ai.ask_gpt(transcript, mode="chat", stream=False, fast=True),
                    self.loop,
                )
                reply = reply_fut.result(timeout=30)
                if reply and self.on_answer:
                    try:
                        self.on_answer(reply)
                    except Exception:
                        pass
            except Exception:
                pass

        except Exception as e:
            self._emit(f"flush_error: {e}")
        finally:
            if wav_path:
                try:
                    os.remove(wav_path)
                except Exception:
                    pass

    def _record_loop(self):
        # ── pick a working input device ───────────────────────────────────────
        use_device = None
        use_sr = self.samplerate
        use_ch = self.channels

        try:
            dd = sd.default.device
            use_device = dd[0] if isinstance(dd, (list, tuple)) else dd
        except Exception:
            use_device = None

        info = None
        try:
            if use_device is not None:
                info = sd.query_devices(use_device)
                if int(info.get("max_input_channels", 0) or 0) == 0:
                    info = None
        except Exception:
            info = None

        if info is None:
            for i, d in enumerate(sd.query_devices()):
                if int(d.get("max_input_channels", 0) or 0) > 0:
                    use_device = i
                    info = d
                    break

        if info:
            use_sr = int(info.get("default_samplerate") or self.samplerate)
            use_ch = min(self.channels, int(info.get("max_input_channels") or 1))

        BLOCK = int(use_sr * 0.5)          # 500 ms per callback block
        SILENCE_END = 3                    # 3 × 500 ms = 1.5 s silence → flush
        MIN_SPEECH = 2                     # 2 × 500 ms = 1 s minimum real speech
        MAX_CHUNKS = 20                    # 20 × 500 ms = 10 s → force-flush

        self._emit(
            f"device_selected: {use_device}  channels={use_ch}  samplerate={use_sr}"
        )

        # ── open InputStream with callback queue ──────────────────────────────
        audio_q: queue.Queue = queue.Queue()

        def _cb(indata, frames, time_info, status):
            audio_q.put(indata.copy())

        try:
            stream = sd.InputStream(
                samplerate=use_sr,
                channels=use_ch,
                dtype="int16",
                device=use_device,
                blocksize=BLOCK,
                callback=_cb,
            )
        except Exception as e:
            self._emit(f"stream_open_error: {e}")
            return

        # ── VAD loop ──────────────────────────────────────────────────────────
        speech_buf = []
        silence_chunks = 0
        chunk_count = 0

        with stream:
            self._emit("stream open — now listening")
            while not self._stop_event.is_set():
                try:
                    chunk = audio_q.get(timeout=0.6)
                except queue.Empty:
                    # timeout counts as silence
                    if speech_buf:
                        silence_chunks += 1
                        if silence_chunks >= SILENCE_END:
                            if len(speech_buf) >= MIN_SPEECH:
                                self._emit(f"flushing {len(speech_buf)} chunk(s) after queue timeout…")
                                self._dispatch_flush(speech_buf[:], use_sr)
                            speech_buf = []
                            silence_chunks = 0
                    continue

                arr = chunk.flatten().astype(np.int16)
                if arr.size == 0:
                    continue

                rms = float(np.sqrt(np.mean((arr.astype("float32") / 32767.0) ** 2)))
                chunk_count += 1

                # heartbeat every 10 chunks (~5 s)
                if chunk_count % 10 == 0:
                    state = "SPEECH" if speech_buf else "silent"
                    self._emit(f"rms {rms:.4f}  thr {self.rms_threshold:.4f}  {state}")

                if rms >= self.rms_threshold:
                    speech_buf.append(arr)
                    silence_chunks = 0
                    if len(speech_buf) >= MAX_CHUNKS:
                        self._emit(f"force-flushing {len(speech_buf)} chunk(s)…")
                        self._dispatch_flush(speech_buf[:], use_sr)
                        speech_buf = []
                        silence_chunks = 0
                else:
                    if speech_buf:
                        silence_chunks += 1
                        if silence_chunks >= SILENCE_END:
                            if len(speech_buf) >= MIN_SPEECH:
                                self._emit(f"flushing {len(speech_buf)} chunk(s)…")
                                self._dispatch_flush(speech_buf[:], use_sr)
                            speech_buf = []
                            silence_chunks = 0

        # flush anything left when stopped
        if speech_buf:
            self._flush(speech_buf[:], use_sr)
