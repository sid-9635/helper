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
    from config import DEFAULT_MIC_DEVICE as CONFIG_DEFAULT_MIC_DEVICE
except Exception:
    CONFIG_DEFAULT_MIC_DEVICE = None

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
        preferred_device: Optional[int] = None,
        selected_model: str = "gpt-4o",
        samplerate: int = 16000,
        channels: int = 1,
        chunk_seconds: float = 0.2,    # 200 ms blocks — fast VAD with low latency
        rms_threshold: float = 0.01,
    ):
        self.ai = ai
        self.loop = loop
        self.on_transcript = on_transcript
        self.on_answer = on_answer
        self.on_answer_delta = None  # set externally for streaming output
        self.on_status = on_status
        self.preferred_device = preferred_device
        self.selected_model = selected_model
        self.samplerate = samplerate
        self.channels = channels
        self.chunk_seconds = max(0.1, chunk_seconds)  # floor at 100 ms
        self.rms_threshold = rms_threshold

        self._thread = None
        self._stop_event = threading.Event()
        self._running = False

    # ── interview filter ──────────────────────────────────────────────────────

    # Set to True to log filter decisions via on_status.
    debug_filter: bool = False

    _INTERVIEW_KEYWORDS = (
        # question words / prompts
        "explain", "what", "why", "how", "difference", "compare", "define",
        "tell me", "can you", "could you", "walk me through", "give me",
        "describe", "when would you", "where would you", "which one",
        "advantages", "disadvantages", "use case", "scenario", "example",
        # engineering actions
        "implement", "write", "code", "design", "approach", "solution",
        "optimize", "improve", "debug", "fix", "issue", "problem",
        "architecture", "system design", "scalability", "performance",
        "test", "testing", "automation", "framework", "api", "database",
        "query", "sql", "graphql", "docker", "kubernetes", "aws",
        # languages / runtimes
        "python", "javascript", "typescript", "java", "golang", "rust",
        "c++", "c#", "ruby", "swift", "kotlin", "php", "bash", "shell",
        # OOP / core concepts
        "class", "object", "method", "function", "variable", "attribute",
        "interface", "abstract", "inherit", "polymorphism", "encapsulation",
        "decorator", "generator", "iterator", "lambda", "closure",
        # data structures / algorithms
        "data", "struct", "list", "dict", "array", "tuple", "set",
        "stack", "queue", "heap", "tree", "graph", "hash", "map",
        "sort", "search", "recursion", "complexity", "big o",
        # concurrency / memory
        "async", "await", "thread", "process", "concurrent", "parallel",
        "memory", "garbage", "pointer", "reference", "scope", "lifetime",
        # web / infra
        "http", "rest", "json", "xml", "microservice", "serverless",
        "cache", "redis", "kafka", "queue", "event", "stream",
    )

    _FILLER_ONLY = {
        "okay", "ok", "hmm", "hm", "yeah", "yep", "yup", "right",
        "so basically", "uh huh", "mhm", "sure", "alright", "alright.",
    }

    def _is_interview_relevant(self, text: str) -> bool:
        """Return True if *text* is worth sending to GPT.

        Passes when any of these conditions hold:
          - contains a question mark
          - contains at least one interview keyword (substring match)

        Skips when:
          - text is under 6 characters after stripping
          - text matches a known filler phrase exactly
        """
        stripped = text.strip()
        if len(stripped) < 6:
            if self.debug_filter:
                self._emit(f"filter:skip (too short) {stripped!r}")
            return False

        lower = stripped.lower()
        if lower.rstrip(".").strip() in self._FILLER_ONLY:
            if self.debug_filter:
                self._emit(f"filter:skip (filler) {stripped!r}")
            return False

        if "?" in stripped:
            if self.debug_filter:
                self._emit(f"filter:pass (question mark) {stripped!r}")
            return True

        for kw in self._INTERVIEW_KEYWORDS:
            if kw in lower:
                if self.debug_filter:
                    self._emit(f"filter:pass (keyword={kw!r}) {stripped!r}")
                return True

        # Length fallback: substantive phrases already past JUNK check are likely real speech
        if len(stripped) >= 20:
            if self.debug_filter:
                self._emit(f"filter:pass (length={len(stripped)}) {stripped!r}")
            return True

        if self.debug_filter:
            self._emit(f"filter:skip (no keyword) {stripped!r}")
        return False

    # ── helpers ──────────────────────────────────────────────────────────────

    def _emit(self, msg: str):
        if self.on_status:
            try:
                self.on_status(msg)
            except Exception:
                pass

    def _valid_input_device(self, device_index: Optional[int]):
        if device_index is None:
            return None
        try:
            info = sd.query_devices(int(device_index))
        except Exception:
            return None
        if int(info.get("max_input_channels", 0) or 0) <= 0:
            return None
        return int(device_index), info

    def _preferred_loopback_device(self):
        preferred_tokens = (
            "stereo mix",
            "what u hear",
            "wave out",
            "loopback",
            "speakers (loopback)",
        )
        try:
            devices = sd.query_devices()
        except Exception:
            return None

        for index, info in enumerate(devices):
            if int(info.get("max_input_channels", 0) or 0) <= 0:
                continue
            name = str(info.get("name", "")).lower()
            if any(token in name for token in preferred_tokens):
                return index, info
        return None

    def _preferred_microphone_device(self):
        preferred_tokens = (
            "microphone array",
            "microphone",
            "mic array",
            "internal mic",
        )
        try:
            devices = sd.query_devices()
        except Exception:
            return None

        for index, info in enumerate(devices):
            if int(info.get("max_input_channels", 0) or 0) <= 0:
                continue
            name = str(info.get("name", "")).lower()
            if any(token in name for token in preferred_tokens):
                return index, info
        return None

    def _pick_input_device(self):
        try:
            default_pair = sd.default.device
            default_input = default_pair[0] if isinstance(default_pair, (list, tuple)) else default_pair
        except Exception:
            default_input = None

        for candidate in (
            self.preferred_device,
            default_input,
            self._preferred_microphone_device(),
            CONFIG_DEFAULT_MIC_DEVICE,
            self._preferred_loopback_device(),
        ):
            if isinstance(candidate, tuple):
                return candidate
            resolved = self._valid_input_device(candidate)
            if resolved is not None:
                return resolved

        try:
            for index, info in enumerate(sd.query_devices()):
                if int(info.get("max_input_channels", 0) or 0) > 0:
                    return index, info
        except Exception:
            pass
        return None, None

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

    def _dispatch_flush(self, buf: list, use_sr: int, partial: bool = False):
        """Run _flush on a throw-away thread so recording continues uninterrupted."""
        threading.Thread(
            target=self._flush, args=(buf, use_sr, partial), daemon=True
        ).start()

    def _flush(self, buf: list, use_sr: int, partial: bool = False):
        """Concatenate buffer → WAV → Whisper → assistant answer.

        When *partial* is True the transcript is prefixed with '[~] ' to signal
        a preliminary result, and the response is not saved to the DB (the final
        flush will do that with the full, more accurate transcript).
        """
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
                if self.on_transcript and not partial:
                    # Only show final transcript in UI — partial is for early GPT only
                    self.on_transcript(transcript)
            except Exception:
                pass

            if not self._is_interview_relevant(transcript):
                self._emit("answer:skipped")  # signal UI to clear Thinking... placeholder
                return

            try:
                if self.on_answer_delta:
                    # ── Streaming path: first token appears in ~0.5 s ─────────────
                    first_token = [True]

                    def _delta(token: str):
                        if first_token[0]:
                            first_token[0] = False
                            self._emit("answer:streaming_start")
                        try:
                            self.on_answer_delta(token)
                        except Exception:
                            pass

                    reply_fut = run_coroutine_threadsafe(
                        self.ai.ask_gpt(transcript, mode="chat", stream=True,
                                        on_delta=_delta, fast=False, selected_model=self.selected_model),
                        self.loop,
                    )
                    reply = reply_fut.result(timeout=30)
                    self._emit("answer:streaming_end")
                    if reply and self.on_answer:
                        try:
                            self.on_answer(reply)
                        except Exception:
                            pass
                else:
                    # ── Streaming request without per-token UI callback ───────────
                    reply_fut = run_coroutine_threadsafe(
                        self.ai.ask_gpt(transcript, mode="chat", stream=True, fast=False, selected_model=self.selected_model),
                        self.loop,
                    )
                    reply = reply_fut.result(timeout=30)
                    self._emit("answer:final")
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
        use_sr = self.samplerate
        use_ch = self.channels
        use_device, info = self._pick_input_device()

        if info:
            use_sr = int(info.get("default_samplerate") or self.samplerate)
            use_ch = min(self.channels, int(info.get("max_input_channels") or 1))

        BLOCK = int(use_sr * self.chunk_seconds)            # samples per callback block
        SILENCE_END    = max(1, round(1.2 / self.chunk_seconds))   # ~1.2 s silence → flush
        MIN_SPEECH     = max(1, round(0.6 / self.chunk_seconds))   # ~0.6 s minimum speech
        MAX_CHUNKS     = max(5, round(10.0 / self.chunk_seconds))  # ~10 s → force-flush
        PARTIAL_CHUNKS = max(MIN_SPEECH + 1, round(3.0 / self.chunk_seconds))  # early partial mark
        HEARTBEAT      = max(1, round(5.0 / self.chunk_seconds))   # rms log every ~5 s
        Q_TIMEOUT      = self.chunk_seconds + 0.1                  # queue.get timeout

        device_name = info.get("name") if info else "unknown"
        self._emit(
            f"device_selected: {use_device} ({device_name})  channels={use_ch}  samplerate={use_sr}"
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
        partial_dispatched = False  # one partial flush per utterance only
        with stream:
            self._emit("stream open — now listening")
            while not self._stop_event.is_set():
                try:
                    chunk = audio_q.get(timeout=Q_TIMEOUT)
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
                            partial_dispatched = False
                    continue

                arr = chunk.flatten().astype(np.int16)
                if arr.size == 0:
                    continue

                rms = float(np.sqrt(np.mean((arr.astype("float32") / 32767.0) ** 2)))
                chunk_count += 1

                # heartbeat every HEARTBEAT chunks (~5 s)
                if chunk_count % HEARTBEAT == 0:
                    state = "SPEECH" if speech_buf else "silent"
                    self._emit(f"rms {rms:.4f}  thr {self.rms_threshold:.4f}  {state}")

                if rms >= self.rms_threshold:
                    speech_buf.append(arr)
                    silence_chunks = 0
                    # Fire one partial flush per utterance once enough speech has accumulated
                    if len(speech_buf) == PARTIAL_CHUNKS and not partial_dispatched:
                        partial_dispatched = True
                        # Only accumulate — do NOT send partial audio to GPT.
                        # Sending 3 s of mid-sentence audio produces "question seems
                        # incomplete" responses. The final flush handles GPT.
                        self._emit(f"partial-mark {len(speech_buf)} chunk(s)")
                    if len(speech_buf) >= MAX_CHUNKS:
                        self._emit(f"force-flushing {len(speech_buf)} chunk(s)…")
                        self._dispatch_flush(speech_buf[:], use_sr)
                        speech_buf = []
                        silence_chunks = 0
                        partial_dispatched = False
                else:
                    if speech_buf:
                        silence_chunks += 1
                        if silence_chunks >= SILENCE_END:
                            if len(speech_buf) >= MIN_SPEECH:
                                self._emit(f"flushing {len(speech_buf)} chunk(s)…")
                                self._dispatch_flush(speech_buf[:], use_sr)
                            speech_buf = []
                            silence_chunks = 0
                            partial_dispatched = False

        # flush anything left when stopped
        if speech_buf:
            self._flush(speech_buf[:], use_sr)
