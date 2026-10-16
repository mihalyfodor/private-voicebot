import asyncio
import base64
import importlib.util
import json
import os
import sys
import threading
import time
import webbrowser
from datetime import datetime

import requests
import uvicorn
from fastapi import FastAPI, Request
from fastapi.responses import HTMLResponse, StreamingResponse

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from tools.gmail import _get_service

_service = None


def get_service():
    global _service
    if _service is None:
        _service = _get_service()
    return _service

OLLAMA_URL = "http://localhost:11434/api/chat"
OLLAMA_MODEL = "gemma4:e2b"
LOG_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "classification.log")
CACHE_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "classification_cache.json")
FEEDBACK_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "classification_feedback.json")
LEARNED_RULES_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "classification_learned_rules.json")
RULE_THRESHOLD = 2
MAX_EMAILS = 50
PORT = 8001

_CLASSIFY_PROMPT = (
    "Classify this email. Reply with ONLY valid JSON, no other text:\n"
    '{{"category": "Action|Info|Ignore", "urgency": "High|Medium|Low", '
    '"confidence": 0-100, "reason": "one sentence", "event_detected": true|false}}\n\n'
    "From: {sender}\nSubject: {subject}\nSnippet: {snippet}"
)

def _load_rules() -> list:
    path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "classification_rules.py")
    if not os.path.exists(path):
        return []
    spec = importlib.util.spec_from_file_location("classification_rules", path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return getattr(mod, "RULES", [])


_RULES = _load_rules()


def apply_rules(sender: str, result: dict) -> dict:
    for rule in _RULES:
        if rule["sender_contains"].lower() in sender.lower() and rule["if"](result):
            result = {**result, **rule["then"]}
    for rule in load_learned_rules():
        if rule["sender"].lower() in sender.lower():
            if "urgency" in rule:
                result["urgency"] = rule["urgency"]
            if "category" in rule:
                result["category"] = rule["category"]
    return result


def maybe_adapt_rules(sender: str) -> bool:
    feedback = load_feedback()
    sender_fb = [f for f in feedback if f["sender"] == sender]
    rules = load_learned_rules()
    rule = next((r for r in rules if r["sender"] == sender), None)
    changed = False

    for field in ("urgency", "category"):
        corrections = [f for f in sender_fb if f["field"] == field]
        if len(corrections) < RULE_THRESHOLD:
            continue
        last = [c["new"] for c in corrections[-RULE_THRESHOLD:]]
        if len(set(last)) == 1:
            if rule is None:
                rule = {"sender": sender}
                rules.append(rule)
            rule[field] = last[0]
            rule["count"] = len(corrections)
            rule["updated"] = datetime.now().isoformat()
            changed = True

    if changed:
        save_learned_rules(rules)
    return changed


_FALLBACK = {
    "category": "Uncategorized",
    "urgency": "Low",
    "confidence": 0,
    "reason": "Classification failed",
    "event_detected": False,
}


def classify(sender: str, subject: str, snippet: str) -> dict:
    prompt = _CLASSIFY_PROMPT.format(sender=sender, subject=subject, snippet=snippet)
    try:
        resp = requests.post(
            OLLAMA_URL,
            json={
                "model": OLLAMA_MODEL,
                "messages": [{"role": "user", "content": prompt}],
                "stream": False,
            },
            timeout=30,
        )
        content = resp.json()["message"]["content"].strip()
        if content.startswith("```"):
            parts = content.split("```")
            content = parts[1] if len(parts) > 1 else parts[0]
            if content.startswith("json"):
                content = content[4:]
        return json.loads(content.strip())
    except Exception:
        return dict(_FALLBACK)


def fetch_emails(service) -> list:
    result = (
        service.users()
        .messages()
        .list(userId="me", labelIds=["INBOX"], maxResults=MAX_EMAILS)
        .execute()
    )
    messages = result.get("messages", [])
    emails = []
    for m in messages:
        msg = (
            service.users()
            .messages()
            .get(
                userId="me",
                id=m["id"],
                format="metadata",
                metadataHeaders=["From", "Subject", "Date"],
            )
            .execute()
        )
        headers = {h["name"]: h["value"] for h in msg["payload"]["headers"]}
        emails.append(
            {
                "id": m["id"],
                "sender": headers.get("From", "Unknown"),
                "subject": headers.get("Subject", "(no subject)"),
                "date": headers.get("Date", ""),
                "snippet": msg.get("snippet", ""),
            }
        )
    return emails


def _decode_part(part) -> tuple:
    mime = part.get("mimeType", "")
    body_data = part.get("body", {}).get("data", "")
    if mime == "text/html" and body_data:
        return base64.urlsafe_b64decode(body_data).decode("utf-8", errors="replace"), ""
    if mime == "text/plain" and body_data:
        return "", base64.urlsafe_b64decode(body_data).decode("utf-8", errors="replace")
    html_parts, text_parts = [], []
    for subpart in part.get("parts", []):
        h, t = _decode_part(subpart)
        if h:
            html_parts.append(h)
        if t:
            text_parts.append(t)
    return "\n".join(html_parts), "\n".join(text_parts)


def fetch_body(service, email_id: str) -> dict:
    msg = (
        service.users()
        .messages()
        .get(userId="me", id=email_id, format="full")
        .execute()
    )
    html, text = _decode_part(msg["payload"])
    return {"id": email_id, "body_html": html, "body_text": text}


def log_result(email_id: str, category: str, urgency: str, confidence: int, latency_ms: int):
    entry = {
        "ts": datetime.now().isoformat(),
        "id": email_id,
        "category": category,
        "urgency": urgency,
        "confidence": confidence,
        "latency_ms": latency_ms,
    }
    with open(LOG_PATH, "a") as f:
        f.write(json.dumps(entry) + "\n")


def load_cache() -> dict:
    if os.path.exists(CACHE_PATH):
        with open(CACHE_PATH) as f:
            return json.load(f)
    return {}


def save_cache(cache: dict):
    with open(CACHE_PATH, "w") as f:
        json.dump(cache, f)


def load_feedback() -> list:
    if os.path.exists(FEEDBACK_PATH):
        with open(FEEDBACK_PATH) as f:
            return json.load(f)
    return []


def save_feedback(entries: list):
    with open(FEEDBACK_PATH, "w") as f:
        json.dump(entries, f, indent=2)


def load_learned_rules() -> list:
    if os.path.exists(LEARNED_RULES_PATH):
        with open(LEARNED_RULES_PATH) as f:
            return json.load(f)
    return []


def save_learned_rules(rules: list):
    with open(LEARNED_RULES_PATH, "w") as f:
        json.dump(rules, f, indent=2)


def _sse(event: str, data: dict) -> str:
    return f"event: {event}\ndata: {json.dumps(data)}\n\n"


app = FastAPI()


@app.get("/", response_class=HTMLResponse)
async def index():
    return DASHBOARD_HTML


@app.get("/stream")
async def stream():
    return StreamingResponse(_classify_stream(), media_type="text/event-stream")


@app.get("/email/{email_id}/body")
async def email_body(email_id: str):
    service = await asyncio.to_thread(get_service)
    return await asyncio.to_thread(fetch_body, service, email_id)


_VALID = {"urgency": {"High", "Medium", "Low"}, "category": {"Action", "Info", "Ignore"}}


@app.post("/feedback/{email_id}")
async def post_feedback(email_id: str, request: Request):
    body = await request.json()
    cache = load_cache()
    if email_id not in cache:
        return {"ok": False, "error": "not in cache"}

    entry = cache[email_id]
    sender = body.get("sender", "")
    feedback = load_feedback()
    updates = {}

    for field in ("urgency", "category"):
        if field in body:
            new = body[field]
            if new not in _VALID[field]:
                continue
            old = entry.get(field, "")
            if old != new:
                entry[field] = new
                updates[field] = new
                feedback.append({
                    "email_id": email_id,
                    "sender": sender,
                    "field": field,
                    "old": old,
                    "new": new,
                    "ts": datetime.now().isoformat(),
                })

    rule_updated = False
    if updates:
        cache[email_id] = entry
        save_cache(cache)
        save_feedback(feedback)
        if sender:
            rule_updated = maybe_adapt_rules(sender)

    return {**updates, "rule_updated": rule_updated}


@app.delete("/cache")
async def clear_all_cache():
    save_cache({})
    return {"ok": True}


@app.delete("/cache/{email_id}")
async def clear_one_cache(email_id: str):
    cache = load_cache()
    cache.pop(email_id, None)
    save_cache(cache)
    return {"ok": True}


async def _classify_stream():
    service = await asyncio.to_thread(get_service)
    emails = await asyncio.to_thread(fetch_emails, service)

    if not emails:
        yield _sse("empty", {})
        return

    yield _sse("count", {"total": len(emails)})

    cache = load_cache()
    counts: dict = {}
    total_confidence = 0
    low_confidence_count = 0
    start_total = time.time()

    for i, email in enumerate(emails):
        cached = email["id"] in cache
        if cached:
            result = cache[email["id"]]
            latency_ms = 0
        else:
            t0 = time.time()
            result = await asyncio.to_thread(
                classify, email["sender"], email["subject"], email["snippet"]
            )
            latency_ms = int((time.time() - t0) * 1000)
            result = apply_rules(email["sender"], result)
            result["ts"] = datetime.now().isoformat()
            cache[email["id"]] = result
            save_cache(cache)

        category = result.get("category", "Uncategorized")
        urgency = result.get("urgency", "Low")
        confidence = int(result.get("confidence", 0))
        reason = result.get("reason", "")
        event_detected = bool(result.get("event_detected", False))

        counts[category] = counts.get(category, 0) + 1
        total_confidence += confidence
        if confidence < 60:
            low_confidence_count += 1

        if not cached:
            log_result(email["id"], category, urgency, confidence, latency_ms)

        yield _sse(
            "card",
            {
                "id": email["id"],
                "sender": email["sender"],
                "subject": email["subject"],
                "date": email["date"],
                "snippet": email["snippet"],
                "category": category,
                "urgency": urgency,
                "confidence": confidence,
                "reason": reason,
                "event_detected": event_detected,
                "latency_ms": latency_ms,
                "cached": cached,
            },
        )

    n = len(emails)
    yield _sse(
        "summary",
        {
            "action_count": counts.get("Action", 0),
            "info_count": counts.get("Info", 0),
            "ignore_count": counts.get("Ignore", 0),
            "avg_confidence": round(total_confidence / n) if n else 0,
            "low_confidence_count": low_confidence_count,
            "total_latency_ms": int((time.time() - start_total) * 1000),
        },
    )


DASHBOARD_HTML = """<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0">
<title>Email Triage</title>
<style>
* { box-sizing: border-box; margin: 0; padding: 0; }
body { font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif; background: #f5f5f5; color: #222; }
header { background: #fff; border-bottom: 1px solid #ddd; padding: 16px 24px; position: sticky; top: 0; z-index: 10; }
header h1 { font-size: 1.2rem; font-weight: 600; }
#progress { font-size: 0.85rem; color: #666; margin-top: 4px; }
#summary-bar { background: #fff; border-bottom: 1px solid #ddd; padding: 10px 24px; display: none; font-size: 0.85rem; gap: 20px; flex-wrap: wrap; align-items: center; }
#cards { padding: 16px 24px; max-width: 860px; margin: 0 auto; }
.card { background: #fff; border: 1px solid #ddd; border-radius: 8px; margin-bottom: 12px; overflow: hidden; }
.card.low-confidence { border: 2px dashed #bbb; }
.card-header { padding: 14px 16px; display: flex; gap: 12px; align-items: flex-start; flex-wrap: wrap; cursor: pointer; }
.card-header:hover { background: #fafafa; }
.card-meta { flex: 1; min-width: 0; }
.card-sender { font-size: 0.78rem; color: #888; white-space: nowrap; overflow: hidden; text-overflow: ellipsis; }
.card-subject { font-weight: 600; font-size: 0.95rem; margin: 3px 0 2px; }
.card-date { font-size: 0.73rem; color: #aaa; }
.card-reason { font-size: 0.82rem; color: #555; margin-top: 5px; }
.badges { display: flex; gap: 6px; flex-wrap: wrap; align-items: flex-start; padding-top: 2px; }
.badge { font-size: 0.71rem; font-weight: 600; padding: 3px 8px; border-radius: 12px; white-space: nowrap; }
.badge-Action { background: #fee2e2; color: #b91c1c; }
.badge-Info { background: #dbeafe; color: #1d4ed8; }
.badge-Ignore { background: #f3f4f6; color: #6b7280; }
.badge-Uncategorized { background: #f3f4f6; color: #9ca3af; }
.badge-High { background: #fef3c7; color: #b45309; }
.badge-Medium { background: #e0f2fe; color: #0369a1; }
.badge-Low { background: #f0fdf4; color: #15803d; }
.badge-conf { background: #f3f4f6; color: #374151; }
.badge-conf.low-conf { background: #f3f4f6; color: #9ca3af; }
.card-body { display: none; padding: 0 16px 14px; border-top: 1px solid #f0f0f0; }
.card-body.open { display: block; }
.email-content { margin-top: 12px; max-height: 420px; overflow-y: auto; border: 1px solid #eee; border-radius: 4px; padding: 12px; background: #fafafa; font-size: 0.88rem; }
.email-content iframe { width: 100%; border: none; display: block; }
.debug-toggle { font-size: 0.74rem; color: #bbb; cursor: pointer; margin-top: 10px; display: inline-block; user-select: none; }
.debug-toggle:hover { color: #888; }
.debug-json { display: none; background: #1e1e1e; color: #d4d4d4; font-family: monospace; font-size: 0.76rem; padding: 10px; border-radius: 4px; margin-top: 6px; overflow-x: auto; white-space: pre; }
.debug-json.open { display: block; }
.empty-msg { color: #888; font-size: 0.95rem; padding: 40px 0; text-align: center; }
.header-row { display: flex; align-items: center; justify-content: space-between; }
.btn-clear-all { font-size: 0.78rem; padding: 5px 12px; border: 1px solid #ddd; border-radius: 6px; background: #fff; color: #666; cursor: pointer; }
.btn-clear-all:hover { background: #fee2e2; border-color: #fca5a5; color: #b91c1c; }
.btn-clear-card { font-size: 0.7rem; padding: 2px 7px; border: 1px solid #e5e7eb; border-radius: 10px; background: #fff; color: #9ca3af; cursor: pointer; line-height: 1.4; }
.btn-clear-card:hover { background: #fee2e2; border-color: #fca5a5; color: #b91c1c; }
.badge-cached { background: #f9fafb; color: #d1d5db; font-style: italic; }
.btn-importance { font-size: 0.65rem; padding: 1px 5px; border: 1px solid #e5e7eb; border-radius: 4px; background: #fff; color: #9ca3af; cursor: pointer; line-height: 1.6; }
.btn-importance:hover { background: #f3f4f6; color: #374151; }
.category-select { font-size: 0.71rem; font-weight: 600; padding: 3px 8px; border-radius: 12px; border: 1px solid transparent; cursor: pointer; outline: none; appearance: none; -webkit-appearance: none; }
.toast { position: fixed; bottom: 24px; right: 24px; background: #1e1e1e; color: #fff; padding: 10px 16px; border-radius: 8px; font-size: 0.82rem; opacity: 0; transition: opacity 0.3s; pointer-events: none; z-index: 100; }
.toast.show { opacity: 1; }
</style>
</head>
<body>
<header>
  <div class="header-row">
    <h1 id="inbox-title">Inbox</h1>
    <button class="btn-clear-all" onclick="clearAll()">Clear all</button>
  </div>
  <div id="progress">Connecting...</div>
</header>
<div id="summary-bar"></div>
<div id="cards"><p class="empty-msg" id="loading-msg">Loading inbox...</p></div>

<script>
const titleEl = document.getElementById('inbox-title');
const progressEl = document.getElementById('progress');
const summaryBarEl = document.getElementById('summary-bar');
const cardsEl = document.getElementById('cards');
const loadingMsg = document.getElementById('loading-msg');

let total = 0, classified = 0;

const es = new EventSource('/stream');

es.addEventListener('count', e => {
  const d = JSON.parse(e.data);
  total = d.total;
  titleEl.textContent = 'Inbox — ' + total + ' email' + (total !== 1 ? 's' : '');
  progressEl.textContent = 'Classified 0 of ' + total + '...';
  if (loadingMsg) loadingMsg.remove();
});

es.addEventListener('empty', () => {
  titleEl.textContent = 'Inbox';
  progressEl.textContent = '';
  cardsEl.innerHTML = '<p class="empty-msg">Inbox empty.</p>';
  es.close();
});

es.addEventListener('card', e => {
  const d = JSON.parse(e.data);
  classified++;
  progressEl.textContent = 'Classified ' + classified + ' of ' + total + '...';

  const lowConf = d.confidence < 60;
  const card = document.createElement('div');
  card.className = 'card' + (lowConf ? ' low-confidence' : '');
  card.dataset.id = d.id;

  const cachedBadge = d.cached ? '<span class="badge badge-cached">cached</span>' : '';
  card.innerHTML =
    '<div class="card-header">' +
      '<div class="card-meta">' +
        '<div class="card-sender">' + esc(d.sender) + '</div>' +
        '<div class="card-subject">' + esc(d.subject) + '</div>' +
        '<div class="card-date">' + esc(d.date) + '</div>' +
        '<div class="card-reason">' + esc(d.reason) + '</div>' +
      '</div>' +
      '<div class="badges">' +
        '<select class="badge category-select badge-' + d.category + '" id="cat-' + d.id + '">' +
          '<option value="Action"' + (d.category === 'Action' ? ' selected' : '') + '>Action</option>' +
          '<option value="Info"' + (d.category === 'Info' ? ' selected' : '') + '>Info</option>' +
          '<option value="Ignore"' + (d.category === 'Ignore' ? ' selected' : '') + '>Ignore</option>' +
        '</select>' +
        '<span class="badge badge-' + d.urgency + '" id="urg-' + d.id + '">' + esc(d.urgency) + '</span>' +
        '<button class="btn-importance" data-id="' + d.id + '" data-dir="1" title="Increase urgency">▲</button>' +
        '<button class="btn-importance" data-id="' + d.id + '" data-dir="-1" title="Decrease urgency">▼</button>' +
        '<span class="badge badge-conf' + (lowConf ? ' low-conf' : '') + '">' + d.confidence + '% confident</span>' +
        cachedBadge +
        '<button class="btn-clear-card" title="Clear cache for this email">↺</button>' +
      '</div>' +
    '</div>' +
    '<div class="card-body" id="body-' + d.id + '">' +
      '<div class="email-content" id="html-' + d.id + '"><em style="color:#aaa">Loading...</em></div>' +
      '<span class="debug-toggle" id="dtog-' + d.id + '">▶ Debug</span>' +
      '<pre class="debug-json" id="debug-' + d.id + '">' + esc(JSON.stringify(d, null, 2)) + '</pre>' +
    '</div>';

  card.querySelector('.card-header').addEventListener('click', () => toggleCard(d.id));
  card.querySelector('.btn-clear-card').addEventListener('click', ev => {
    ev.stopPropagation();
    clearOne(d.id);
  });
  card.querySelector('.debug-toggle').addEventListener('click', ev => {
    ev.stopPropagation();
    const dbg = document.getElementById('debug-' + d.id);
    const tog = document.getElementById('dtog-' + d.id);
    dbg.classList.toggle('open');
    tog.textContent = dbg.classList.contains('open') ? '▼ Debug' : '▶ Debug';
  });
  card.querySelector('#cat-' + d.id).addEventListener('change', ev => {
    ev.stopPropagation();
    const sel = ev.target;
    const newCat = sel.value;
    const oldCat = sel.dataset.current || d.category;
    sel.dataset.current = newCat;
    sel.className = 'badge category-select badge-' + newCat;
    sendFeedback(d.id, d.sender, {category: newCat}, () => {}, () => {
      sel.value = oldCat;
      sel.dataset.current = oldCat;
      sel.className = 'badge category-select badge-' + oldCat;
    });
  });
  card.querySelectorAll('.btn-importance').forEach(btn => {
    btn.addEventListener('click', ev => {
      ev.stopPropagation();
      adjustUrgency(d.id, d.sender, parseInt(btn.dataset.dir));
    });
  });

  cardsEl.prepend(card);
});

es.addEventListener('summary', e => {
  const d = JSON.parse(e.data);
  summaryBarEl.style.display = 'flex';
  summaryBarEl.innerHTML =
    '<span>Action: <strong>' + d.action_count + '</strong></span>' +
    '<span>Info: <strong>' + d.info_count + '</strong></span>' +
    '<span>Ignore: <strong>' + d.ignore_count + '</strong></span>' +
    '<span>Avg confidence: <strong>' + d.avg_confidence + '%</strong></span>' +
    '<span>Low confidence: <strong>' + d.low_confidence_count + '</strong></span>' +
    '<span>Total time: <strong>' + (d.total_latency_ms / 1000).toFixed(1) + 's</strong></span>';
  progressEl.textContent = 'Done — ' + total + ' emails classified.';
  es.close();
});

es.onerror = () => {
  progressEl.textContent = 'Stream error — check server.';
};

function clearAll() {
  fetch('/cache', {method: 'DELETE'}).then(() => location.reload());
}

function clearOne(id) {
  fetch('/cache/' + id, {method: 'DELETE'}).then(() => location.reload());
}

function toggleCard(id) {
  const bodyEl = document.getElementById('body-' + id);
  const opening = !bodyEl.classList.contains('open');
  bodyEl.classList.toggle('open', opening);
  if (opening) loadBody(id);
}

function loadBody(id) {
  const htmlEl = document.getElementById('html-' + id);
  if (htmlEl.dataset.loaded) return;
  htmlEl.dataset.loaded = '1';
  fetch('/email/' + id + '/body')
    .then(r => r.json())
    .then(data => {
      if (data.body_html) {
        const iframe = document.createElement('iframe');
        iframe.srcdoc = data.body_html;
        iframe.style.cssText = 'width:100%;border:none;min-height:120px;display:block';
        iframe.onload = () => {
          try { iframe.style.height = iframe.contentDocument.body.scrollHeight + 30 + 'px'; } catch(e) {}
        };
        htmlEl.innerHTML = '';
        htmlEl.appendChild(iframe);
      } else if (data.body_text) {
        htmlEl.innerHTML = '<pre style="white-space:pre-wrap;font-size:0.85rem">' + esc(data.body_text) + '</pre>';
      } else {
        htmlEl.textContent = '(no body)';
      }
    })
    .catch(() => { htmlEl.textContent = 'Failed to load body.'; });
}

function esc(s) {
  return String(s)
    .replace(/&/g, '&amp;')
    .replace(/</g, '&lt;')
    .replace(/>/g, '&gt;')
    .replace(/"/g, '&quot;');
}

const URGENCY = ['Low', 'Medium', 'High'];

function adjustUrgency(id, sender, dir) {
  const el = document.getElementById('urg-' + id);
  const cur = URGENCY.indexOf(el.textContent.trim());
  const next = Math.max(0, Math.min(2, cur + dir));
  if (next === cur) return;
  const newUrg = URGENCY[next];
  sendFeedback(id, sender, {urgency: newUrg}, () => {
    el.className = 'badge badge-' + newUrg;
    el.textContent = newUrg;
  }, () => {});
}

function sendFeedback(id, sender, updates, cb, onErr) {
  fetch('/feedback/' + id, {
    method: 'POST',
    headers: {'Content-Type': 'application/json'},
    body: JSON.stringify({sender, ...updates})
  })
  .then(r => { if (!r.ok) throw new Error(r.status); return r.json(); })
  .then(data => {
    if (data.rule_updated) {
      const name = sender.replace(/<.*>/, '').trim() || sender;
      showToast('Rule updated for ' + name);
    }
    if (cb) cb(data);
  })
  .catch(() => { if (onErr) onErr(); });
}

function showToast(msg) {
  let t = document.getElementById('toast');
  if (!t) {
    t = document.createElement('div');
    t.id = 'toast';
    t.className = 'toast';
    document.body.appendChild(t);
  }
  t.textContent = msg;
  t.classList.add('show');
  clearTimeout(t._timer);
  t._timer = setTimeout(() => t.classList.remove('show'), 3000);
}
</script>
</body>
</html>"""


def _open_browser():
    time.sleep(0.8)
    webbrowser.open(f"http://localhost:{PORT}")


if __name__ == "__main__":
    threading.Thread(target=_open_browser, daemon=True).start()
    try:
        uvicorn.run(app, host="0.0.0.0", port=PORT, log_level="warning")
    except KeyboardInterrupt:
        pass
