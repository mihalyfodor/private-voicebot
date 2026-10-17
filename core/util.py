"""Small shared helpers: the clock every stored date reads, the token estimate, the running-app probe."""
import os
from datetime import date, datetime

def _today() -> date:
    """Today. The single source of "now" for every date the app stores — the eval harness
    monkeypatches it to run a scripted conversation against a shifted clock."""
    return date.today()


def _today_iso() -> str:
    """Today as ``YYYY-MM-DD`` — the form every date stored in the profile takes."""
    return _today().isoformat()


def tokens_est(text: str) -> int:
    return len(text) // 4


def running_app() -> str:
    """"app is running — …" when the app answers on prod (8010), dev (8011) or `PORT`; "" otherwise.
    Also the guard of `python -m vault reindex` (PRD 53)."""
    import urllib.error
    import urllib.request

    ports = {os.getenv("PROD_PORT") or "8010", os.getenv("DEV_PORT") or "8011", os.getenv("PORT") or ""} - {""}
    for port in sorted(ports):
        try:
            urllib.request.urlopen(f"http://127.0.0.1:{port}/", timeout=2).close()
        except urllib.error.HTTPError:
            pass                                # an error page is still an app answering
        except (OSError, ValueError):
            continue
        return f"app is running — scripts/voicebot stop (port {port})"
    return ""


def _stamp(time_format: str) -> str:
    """An ISO timestamp whose date part comes from :func:`_today` (so a mocked clock shows)."""
    return f"{_today_iso()}T{datetime.now().strftime(time_format)}"
