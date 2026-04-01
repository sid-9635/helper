# Helper

Quick start:

1. Copy `.env.example` to `.env` and fill `OPENAI_API_KEY`.
2. (Recommended) create and activate your virtualenv:

```powershell
python -m venv .venv
& .venv\Scripts\Activate.ps1
pip install -r requirements.txt
```

3. Run the app:

```powershell
& .venv\Scripts\python.exe main.py
```

Notes:
- This repository uses `config.py` to load `OPENAI_API_KEY` from the environment or a local `.env` file.
- Do NOT commit your real `.env` to source control.
# Stealth Chat Assistant (overlay)

A compact, always-on-top Tkinter overlay that captures audio or screen content, sends it to Whisper / GPT, and shows streaming assistant responses. Designed for rapid coding interview help and short, actionable answers.

This README covers full project setup, dependencies, Tesseract OCR integration, quick usage, troubleshooting, and developer notes.

---

## Features
- Transparent overlay window with small footprint
- Live audio recording (WASAPI loopback / system audio) and Whisper transcription
- GPT chat with streaming token-by-token UI updates
- Screen capture + OCR (pytesseract) -> AI pipeline
- Response area supports selection and copy
- Capture-protection heuristics on Windows (SetWindowDisplayAffinity)
- SQLite conversation history stored locally

---

## Requirements
- Windows 10/11 (features use Windows-specific APIs for capture protection)
- Python 3.10+ (project uses virtualenv in `.venv`)
- Tesseract OCR (for screen OCR feature) — see `TESSERACT_INSTALLATION.md`
- OpenAI API key (set `OPENAI_API_KEY` or edit `config.py`)

Python packages are listed in `requirements.txt`.

---

## Quick setup (recommended)

1. Clone repository and open terminal in project root.

2. Create and activate venv (Windows PowerShell example):

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
```

3. Install Python dependencies:

```powershell
python -m pip install -r requirements.txt
```

4. Install Tesseract OCR (if you plan to use the `Capture` OCR feature):

5. Put your OpenAI API key in one of these places:

6. Run the app:

```powershell
python main.py
```

The overlay should appear. Use the prompt box to type queries or use the Start/Stop buttons for audio capture.

---

## Using the Capture (OCR) feature

1. Ensure Tesseract is installed and `tesseract --version` works from a terminal.
2. Start the app and click the `Capture` button.
   - The app will take a full-screen screenshot, run local OCR, and if text is detected will send it to GPT as a coding prompt.
   - For speed and accuracy, you can crop the captured image (region-select feature planned). The app currently captures full screen by default.

Notes:
- If `pytesseract` is not installed, the app shows a helpful message.
- If you installed Tesseract but it's not on PATH, set the path in Python code:

```python
import pytesseract
pytesseract.pytesseract.tesseract_cmd = r"C:\Program Files\Tesseract-OCR\tesseract.exe"
```

---

## UI behavior and conventions
- Responses are streamed token-by-token into the overlay.
- When a coding prompt is detected, the assistant is instructed to return code-only answers wrapped in a single fenced code block (no explanation) to make copying easy.
- The response area remains anchored at the start of each assistant stream; manual scrolling cancels auto-anchoring so you can read long responses.

---

## Files and structure
- `main.py` — small entrypoint that runs the UI
- `ui.py` — overlay UI (moved out of `main.py` during refactor)
- `ai_bridge.py` — handles transcription and GPT calls (streaming support)
- `recorder.py` — audio capture utilities (WASAPI / sounddevice)
- `database.py` — SQLite message history
- `config.py` — simple config (OpenAI API key placeholder)
- `requirements.txt` — Python dependencies
- `TESSERACT_INSTALLATION.md` — Tesseract installation notes (Windows)

---

## Developer notes
- The assistant system prompt in `ai_bridge.py` enforces code-only responses for coding prompts. Modify only if you need a different behavior.
- The streaming logic uses aiohttp and reads server-sent events (`data: ...`) chunks; keep `ask_gpt(..., stream=True)` semantics when calling from other code.
- To add a region-select capture tool, implement a transparent top-level window that captures mouse drag coordinates and then pass the cropped PIL image to `pytesseract.image_to_string` (I can add this on request).

---

## Troubleshooting
- Missing `aiohttp` / other packages: run `python -m pip install -r requirements.txt`.
- `ModuleNotFoundError: No module named 'pytesseract'` or `PIL`: install `Pillow` and `pytesseract`.
- `TesseractNotFoundError`: ensure Tesseract binary is installed and in PATH, or set `pytesseract.pytesseract.tesseract_cmd`.
- If OpenAI requests fail: verify `OPENAI_API_KEY` and network connectivity.

---

## Next improvements (suggested)
- Add region selection capture (fast, recommended)
- Add a small "Scroll to end" button in the UI when auto-anchoring is disabled
- Allow image-only prompt path (send image directly to a Vision endpoint / cloud OCR) as fallback
- Add unit tests around `ai_bridge` network handling and `database` storage

---

If you'd like, I can now:
- Install `Pillow` and `pytesseract` into the venv; and
- Implement the region-selection capture (fast path) so the OCR and AI latency drops significantly.

Tell me which to run next.
