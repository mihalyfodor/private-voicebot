# 03 — Email Classification Dashboard

**Status:** `implementation`

## Overview

Standalone CLI script (`workflows/email_classification.py`) that fetches all current inbox emails and streams them to a browser dashboard one-by-one as each is classified by local Ollama LLM. Each card shows category (Action / Info / Ignore), urgency (High / Medium / Low), confidence %, and LLM reasoning. Clicking a card reveals the full email body. Users can correct classifications inline; corrections feed a learned-rules system that adapts future runs.

Phased implementation:
- **Phase 1 (this PRD):** Read-only triage dashboard with progressive streaming + inline correction + rule learning. No re-auth needed.
- **Phase 2 (future PRD):** Actions — archive email, create Google Calendar event.

## User Stories

1. Run `python3 workflows/email_classification.py` → browser opens showing empty dashboard with "Loading inbox..."
2. Cards appear one at a time as each email is classified — can read earlier cards while later ones still processing
3. Click a card → full email body loads inline to help decide what to do
4. All emails done → summary shown: "X Action, Y Info, Z Ignore — avg confidence 81%, 2 low-confidence"
5. Re-run → browser re-opens, fresh classification of current inbox state
6. Click ▲/▼ buttons next to an urgency badge → urgency updates instantly (Low↔Medium↔High)
7. Click a category badge → dropdown appears; select new value → badge updates instantly
8. After correcting the same sender twice → toast "Rule updated for [Name]"; future runs auto-apply the correction for that sender

## UI/UX

- FastAPI server on port 8001, auto-opens browser via `webbrowser.open()`
- Page loads immediately with loading state
- **Streaming via SSE:** as each email is classified, server emits an event → JS appends card to list
- **Card (collapsed):** sender, subject, date, category badge (color-coded, clickable dropdown), urgency badge + ▲▼ buttons, confidence % (e.g. "87% confident"), LLM reason (1 sentence)
- **Card (expanded, on click):** full email body rendered (HTML preferred, plaintext fallback) + collapsible "Debug" section showing raw LLM JSON
- Low-confidence cards (< 60%) — dashed border + grey badge visual cue
- **Inline correction:** clicking ▲/▼ next to urgency badge shifts Low→Medium→High (or back); clicking category badge opens dropdown (Action / Info / Ignore); both call `POST /feedback/{id}` and update in place
- **Toast notification:** appears bottom-right when a correction triggers a new or updated learned rule: "Rule updated for [Sender Name]"
- Progress counter during streaming: "Classified 3 of 12..."
- Summary bar fills in at completion: counts per category/urgency, avg confidence, total processing time
- Header: "Inbox — X emails" (count known upfront)
- No Archive / Add to Calendar buttons in Phase 1

## Technical Approach

**Stack:** Python, FastAPI, vanilla JS, Gmail API, Ollama (local LLM)

**Key decisions:**

| # | Decision | Rationale |
|---|----------|-----------|
| 1 | `workflows/` folder | Standalone scripts, separate from `tools/` (which are LLM-callable) |
| 2 | FastAPI on port 8001 | Consistent with existing stack (chatbot.py uses FastAPI on 8000) |
| 3 | SSE (`text/event-stream`) | Simpler than WebSocket for one-way server→client push; FastAPI `StreamingResponse` with async generator |
| 4 | One LLM call per email | Enables progressive streaming; batch would require all emails to finish before showing anything |
| 5 | Full email body on demand | Fetch via `/email/{id}/body` only on card click — avoids slow startup for large inboxes |
| 6 | Import `tools/gmail.py._get_service()` | Auth logic already exists; no duplication; readonly scope covers Phase 1 |
| 7 | LLM input per email: `{sender, subject, snippet}` | Enough signal for classification; body too verbose for batch; model is `gemma4:e2b` (same as chatbot) |
| 8 | Confidence from LLM | Prompt requests `confidence` (0–100 int); displayed on card + logged; low-confidence flagged visually |
| 9 | Structured observability log | JSON lines to `workflows/triage.log` (gitignored) — id, category, urgency, confidence, latency_ms per email |
| 10 | Cards in arrival order (date desc) | Natural inbox order; no re-sorting during stream; simpler |
| 11 | Server runs until Ctrl+C | Processing finishes, browser stays open for reading; user kills manually |
| 12 | Feedback stored in `classification_feedback.json` (gitignored) | Append-only; each entry: email_id, sender, field, old, new, ts |
| 13 | Learned rules in `classification_learned_rules.json` (gitignored) | JSON list; each rule: sender, urgency?, category?, count, updated — applied after lambda rules |
| 14 | Rule threshold = 2 | Two same-direction corrections for same sender → create/update rule; avoids noise from one-offs |
| 15 | `apply_rules()` loads learned rules fresh each call | Rules can be written by feedback endpoint mid-session; must see new rules immediately |

**Risks / mitigations:**
- LLM JSON output malformed → validate + fallback to `{category: "Uncategorized", urgency: "Low", confidence: 0, reason: "Classification failed"}`
- Large inbox → slow total time, but UX fine because cards stream progressively; cap at 50 emails
- MIME parsing → prefer `text/html`, fallback `text/plain`; log warnings for unrecognised parts

## Data Model / State

No persistent state file — inbox is the source of truth.

```python
# SSE event payload (one per email, streamed)
{
  "id": "18f3a...",
  "sender": "teacher@kindergarten.hu",
  "subject": "Sports Day on Friday",
  "date": "2026-05-02T08:30:00",
  "snippet": "...",
  "category": "Action",     # Action | Info | Ignore
  "urgency": "High",        # High | Medium | Low
  "confidence": 87,         # 0–100, from LLM
  "reason": "Event needs RSVP by Thursday",
  "event_detected": true,   # hint for Phase 2 calendar action
  "latency_ms": 412         # server-side LLM call duration
}

# Full body (fetched on demand)
{
  "id": "18f3a...",
  "body_html": "...",   # prefer HTML
  "body_text": "..."    # fallback
}

# Summary SSE event (final, type="summary")
{
  "type": "summary",
  "action_count": 1,
  "info_count": 3,
  "ignore_count": 3,
  "avg_confidence": 81,
  "low_confidence_count": 2,
  "total_latency_ms": 4823
}

# Feedback entry (appended to classification_feedback.json)
{
  "email_id": "18f3a...",
  "sender": "teacher@kindergarten.hu",
  "field": "urgency",           # "urgency" | "category"
  "old": "Medium",
  "new": "High",
  "ts": "2026-05-02T09:00:00"
}

# Learned rule (entry in classification_learned_rules.json)
{
  "sender": "teacher@kindergarten.hu",
  "urgency": "High",            # optional
  "category": "Action",         # optional
  "count": 2,
  "updated": "2026-05-02T09:01:00"
}
```

## API Endpoints

| Method | Path | Purpose |
|--------|------|---------|
| GET | `/` | Serve dashboard HTML |
| GET | `/stream` | SSE: classify emails one-by-one, emit card events |
| GET | `/email/{id}/body` | Fetch full email body on demand |
| POST | `/feedback/{id}` | Update urgency/category; persist feedback; adapt rules |
| DELETE | `/cache` | Clear all cached classifications |
| DELETE | `/cache/{id}` | Clear one email's cached classification |

`POST /feedback/{id}` request body:
```json
{"sender": "...", "urgency": "High", "category": "Action"}
```
Both `urgency` and `category` are optional — send only what changed. Response:
```json
{"urgency": "High", "rule_updated": true}
```

## Test Scenarios

1. **Dashboard loads** — run script → browser opens, page shows "Loading..." and progress counter immediately
2. **Cards stream** — emails appear one at a time, each with category/urgency/confidence/reason badges
3. **Summary correct** — after all cards load, summary counts match actual email classifications
4. **Click card** — body loads inline; click again → collapses; Debug section shows raw LLM JSON
5. **Empty inbox** — no emails → shows "Inbox empty" immediately, no SSE events
6. **LLM down** — Ollama not running → emails still load (metadata only) with "Classification unavailable" badge; no crash
7. **Large inbox** — 20+ emails → all stream without timeout or crash
8. **Low confidence** — LLM returns confidence < 60% → card shows dashed border; summary increments `low_confidence_count`
9. **Observability log** — after run, `workflows/triage.log` contains one JSON line per email with id, category, urgency, confidence, latency_ms
10. **Urgency correction** — click ▲ on Medium card → badge updates to High; `classification_cache.json` and `classification_feedback.json` reflect new value
11. **Category correction** — click category badge → dropdown; select different value → badge updates; feedback recorded
12. **Rule creation** — correct same sender's urgency ▲ twice → toast "Rule updated for [Name]"; `classification_learned_rules.json` has entry for that sender
13. **Rule applied on re-run** — clear cache; re-run → sender from step 12 gets corrected urgency without manual intervention
14. **Rule threshold not hit** — correct sender once → no toast, no rule written

## Out of Scope (Phase 1)

- Archive / Gmail write actions (Phase 2)
- Google Calendar integration (Phase 2)
- OAuth scope expansion (Phase 2)
- Voice trigger or spoken output
- Reply / compose
- Multi-account Gmail
- Push notifications / scheduled auto-run
- Mobile-optimized UI
