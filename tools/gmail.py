import os

from google.oauth2.credentials import Credentials
from google_auth_oauthlib.flow import InstalledAppFlow
from google.auth.transport.requests import Request
from googleapiclient.discovery import build

from untrusted import untrusted

#: the Google endpoints the client library talks to (docs/33-untrusted-text.md read-only policy)
HOSTS = ("gmail.googleapis.com", "oauth2.googleapis.com", "accounts.google.com")

GMAIL_SCOPES = ["https://www.googleapis.com/auth/gmail.readonly"]
# GMAIL_DIR env: where credentials.json / token.json live (default: the working directory;
# the Docker prod mounts them at /data/secrets — docs/40-docker-prod.md)
GMAIL_DIR = os.getenv("GMAIL_DIR") or "."
GMAIL_CREDENTIALS_PATH = os.path.join(GMAIL_DIR, "credentials.json")
GMAIL_TOKEN_PATH = os.path.join(GMAIL_DIR, "token.json")

DEFINITION = {
    "type": "function",
    "function": {
        "name": "get_emails",
        "description": "Returns a summary of the user's recent Gmail inbox emails — sender, subject, and snippet.",
        "parameters": {
            "type": "object",
            "properties": {
                "max_results": {
                    "type": "integer",
                    "description": "Number of emails to fetch. Defaults to 5."
                }
            },
            "required": [],
        },
    },
}


#: the last `get_emails` result, id -> {from, subject, snippet}, in inbox order.
#: `file_source("email", ref)` resolves a spoken ref ("the one from Anna") against this.
LAST_RESULTS: dict = {}


def resolve_ref(ref) -> str | None:
    """A message id, a 1-based index or a substring of sender/subject -> the message id."""
    text = str(ref or "").strip()
    if not text:
        return None
    if text in LAST_RESULTS:
        return text
    if text.isdigit():
        ids = list(LAST_RESULTS)
        index = int(text) - 1
        return ids[index] if 0 <= index < len(ids) else None
    needle = text.lower()
    for msg_id, meta in LAST_RESULTS.items():
        if needle in f"{meta.get('from', '')} {meta.get('subject', '')}".lower():
            return msg_id
    return None


def _body_text(payload: dict) -> str:
    """The first text/plain part of a `format=full` payload (html stripped as a fallback)."""
    import base64
    import re

    plain, html = "", ""
    stack = [payload or {}]
    while stack:
        part = stack.pop(0)
        stack.extend(part.get("parts") or [])
        data = (part.get("body") or {}).get("data")
        if not data:
            continue
        try:
            text = base64.urlsafe_b64decode(data + "==").decode("utf-8", errors="replace")
        except Exception:  # noqa: BLE001
            continue
        if part.get("mimeType") == "text/plain" and not plain:
            plain = text
        elif part.get("mimeType") == "text/html" and not html:
            html = text
    return plain.strip() or " ".join(re.sub(r"<[^>]+>", " ", html).split())


def fetch_message(msg_id: str) -> str:
    """One message as markdown: a From/To/Date/Subject block plus the plain-text body.

    This is what `file_source("email", ...)` stores under `raw/email/<id>.md`.
    """
    service = _get_service()
    msg = service.users().messages().get(userId="me", id=msg_id, format="full").execute()
    payload = msg.get("payload") or {}
    headers = {h["name"]: h["value"] for h in payload.get("headers", [])}
    block = "\n".join(f"{key}: {headers.get(key, '')}" for key in ("From", "To", "Date", "Subject")
                       if headers.get(key))
    return f"{block}\n\n{_body_text(payload) or msg.get('snippet', '')}\n"


def _get_service():
    creds = None
    if os.path.exists(GMAIL_TOKEN_PATH):
        creds = Credentials.from_authorized_user_file(GMAIL_TOKEN_PATH, GMAIL_SCOPES)
    if not creds or not creds.valid:
        if creds and creds.expired and creds.refresh_token:
            creds.refresh(Request())
        else:
            flow = InstalledAppFlow.from_client_secrets_file(GMAIL_CREDENTIALS_PATH, GMAIL_SCOPES)
            creds = flow.run_local_server(port=0)
        with open(GMAIL_TOKEN_PATH, "w") as f:
            f.write(creds.to_json())
    return build("gmail", "v1", credentials=creds)


def is_configured() -> bool:
    return os.path.exists(GMAIL_CREDENTIALS_PATH) or os.path.exists(GMAIL_TOKEN_PATH)


def run(args):
    if not is_configured():
        return ("Email is not set up on this machine: credentials.json is missing. "
                "Tell the user email access is not configured.")
    max_results = args.get("max_results", 5)
    service = _get_service()
    result = service.users().messages().list(
        userId="me", labelIds=["INBOX"], maxResults=max_results
    ).execute()
    messages = result.get("messages", [])
    if not messages:
        return "The inbox is empty."
    lines = []
    LAST_RESULTS.clear()
    for m in messages:
        msg = service.users().messages().get(
            userId="me", id=m["id"], format="metadata",
            metadataHeaders=["From", "Subject"]
        ).execute()
        headers = {h["name"]: h["value"] for h in msg["payload"]["headers"]}
        sender = headers.get("From", "Unknown")
        subject = headers.get("Subject", "(no subject)")
        snippet = msg.get("snippet", "")[:100]
        LAST_RESULTS[m["id"]] = {"from": sender, "subject": subject, "snippet": snippet}
        lines.append(f"From {sender}: {subject}. {snippet}")
    return "Read out these emails naturally: " + untrusted(" | ".join(lines), "Gmail inbox")
