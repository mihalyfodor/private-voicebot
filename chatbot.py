"""Entry point: `python chatbot.py` starts the voicebot (Dockerfile CMD, scripts/voicebot). The code lives in server/ (PRD 50)."""
from dotenv import load_dotenv

load_dotenv()
from core import trace  # noqa: E402

if __name__ == "__main__":
    trace.configure()      # PRD 49: a dev process never starts against prod data; before anything loads

from server import app, ingest, ws  # noqa: E402,F401  (ingest and ws register their routes on `app`)

if __name__ == "__main__":
    app.main()
