import os

# Try to auto-load a local .env if python-dotenv is available.
try:
    from dotenv import load_dotenv
    load_dotenv()
except Exception:
    pass

# Read API key from environment or .env fallback
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")

# If not set, try a simple .env parser in the project root
if not OPENAI_API_KEY:
    env_path = os.path.join(os.path.dirname(__file__), ".env")
    if os.path.exists(env_path):
        try:
            with open(env_path, "r", encoding="utf-8") as fh:
                for raw in fh:
                    line = raw.strip()
                    if not line or line.startswith("#") or "=" not in line:
                        continue
                    k, v = line.split("=", 1)
                    k = k.strip()
                    v = v.strip().strip('"').strip("'")
                    if k == "OPENAI_API_KEY" and v:
                        OPENAI_API_KEY = v
                        os.environ["OPENAI_API_KEY"] = v
                        break
        except Exception:
            pass

# Default tesseract command - adjust if Tesseract is installed elsewhere
TESSERACT_CMD = r"C:\Program Files\Tesseract-OCR\tesseract.exe"

# Default microphone device index (optional)
DEFAULT_MIC_DEVICE = None
import os

# If python-dotenv is available, load a local .env file into the environment.
# This makes it convenient for local testing without committing secrets.
try:
	from dotenv import load_dotenv
	load_dotenv()
except Exception:
	pass

# Load OpenAI API key from environment. Do NOT store secrets in this file.
# Set environment variable `OPENAI_API_KEY` or place it in a local, ignored `.env` file.
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")

# If key wasn't loaded from the environment and a .env file exists, try a simple local parse
# so this works even if `python-dotenv` is not installed.
if not OPENAI_API_KEY:
	env_path = os.path.join(os.path.dirname(__file__), ".env")
	if os.path.exists(env_path):
		try:
			with open(env_path, "r", encoding="utf-8") as fh:
				for raw in fh:
					line = raw.strip()
					if not line or line.startswith("#") or "=" not in line:
						continue
					k, v = line.split("=", 1)
					k = k.strip()
					v = v.strip().strip('"').strip("'")
					if k == "OPENAI_API_KEY" and v:
						OPENAI_API_KEY = v
						os.environ["OPENAI_API_KEY"] = v
						break
		except Exception:
			pass
# Optional: set full path to tesseract executable if it's not on PATH
# Example (Windows): r"C:\Program Files\Tesseract-OCR\tesseract.exe"
# Set to your installed Tesseract path:
TESSERACT_CMD = r"C:\Program Files\Tesseract-OCR\tesseract.exe"
# Optional: default input device index to prefer for live listening (set to None to auto-detect)
# Example: DEFAULT_MIC_DEVICE = 13
DEFAULT_MIC_DEVICE = 13