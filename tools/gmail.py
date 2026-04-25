from google.oauth2.credentials import Credentials
from google_auth_oauthlib.flow import InstalledAppFlow
from google.auth.transport.requests import Request
from googleapiclient.discovery import build

GMAIL_SCOPES = ["https://www.googleapis.com/auth/gmail.readonly"]
GMAIL_CREDENTIALS_PATH = "credentials.json"
GMAIL_TOKEN_PATH = "token.json"

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


def _get_service():
    creds = None
    import os
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


def run(args):
    max_results = args.get("max_results", 5)
    service = _get_service()
    result = service.users().messages().list(
        userId="me", labelIds=["INBOX"], maxResults=max_results
    ).execute()
    messages = result.get("messages", [])
    if not messages:
        return "The inbox is empty."
    lines = []
    for m in messages:
        msg = service.users().messages().get(
            userId="me", id=m["id"], format="metadata",
            metadataHeaders=["From", "Subject"]
        ).execute()
        headers = {h["name"]: h["value"] for h in msg["payload"]["headers"]}
        sender = headers.get("From", "Unknown")
        subject = headers.get("Subject", "(no subject)")
        snippet = msg.get("snippet", "")[:100]
        lines.append(f"From {sender}: {subject}. {snippet}")
    return "Read out these emails naturally: " + " | ".join(lines)
