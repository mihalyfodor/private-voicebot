Status: exploration

# 52 — Approvals: confirm writes by voice or card button

## Overview
Destructive or outbound writes have no common confirm today: Forget is a browser `confirm()` dialog plus an ad-hoc
`confirm: true` flag (static/app.js:1710-1713, chatbot.py:1109-1114), and PRD 47b's "send that to my reMarkable" has
nothing yet (docs/47-remarkable-pull.md:106). This PRD adds one pending approval per session that the user resolves
by saying "yes" or pressing Approve/Reject on the run card (PRD 51). The approval is bound to exactly what was shown;
voice stays first, the button is never required.

## User stories
- As a user, I say "forget the Morrowind guide"; Wanko says what would go, a card shows the pages and memory
  entries it would remove with **Approve** / **Reject**, and nothing is deleted yet.
- As a user, I say "yes" and it is deleted; or I press Approve and get the same result.
- As a user, I say "no" or simply ask something else, and nothing happens; a later click on the old card says
  "that request expired".
- As a user, I say "send that to my reMarkable" (PRD 47b) and approve the push the same way.

## UI/UX
- Awaiting card: the PRD 51 run card with status `awaiting`, the proposal body (forget: page, raw file, pages whose
  notes are cleaned, memory entries; push: card title and page count) and two `actions`: Approve, Reject.
- No modal, no countdown: the card waits for the next turn; the next turn decides.
- Spoken proposal: one line ending in a question ("That removes the guide, its raw file and 3 notes. Go ahead?").
- On resolve: `card_update` drops the buttons and sets status `done` (approved) or `rejected` (a status 52 adds to 51's set); Wanko speaks one line
  ("Forgotten." / "Left it."). Buttons disable on click until the `card_update` or an `error` arrives.
- Run page: while pending it shows the proposal; after approval it shows the commit (hash + message).
- Sources view Forget button: no more `confirm()`; it starts the same proposal (card in the transcript).

## Technical approach
- **State** (`session.py`): `Session.pending = {run_id, kind, proposal, hash, created}` or None, next to the
  per-connection fields (session.py:29-35). `run_id` is the proposing turn's run (`<pid>-<turn_id>`, trace.py:296-302).
- **Propose**: a handler for `kind` returns a read-only proposal (no write): `sources.forget_plan(sha)` (the dry run
  of `sources.forget`, sources.py:1411-1444: page, raw, cleaned pages, memory entries) and `notebooks.push_plan(card)`
  (47b). `hash` = sha256 of the proposal's canonical JSON. The turn sends the awaiting card, speaks the line, sets
  `sess.pending` (replacing any older one).
- **One-turn pop**, the `_deeper_offer` model (llm.py:295, popped at llm.py:1064-1066, re-set at llm.py:1220): every
  turn starts with `pending, sess.pending = sess.pending, None`. In `_respond` (chatbot.py:573), before the
  new-topic / write-up branches: pending ∧ `llm.is_approval(text)` (new; reuses `is_affirmative`'s leads, llm.py:619-623, with
  the write words below) → `resolve_pending(sess, "approve", via="voice"|"typed")`; pending ∧ a short explicit "no" →
  `resolve_pending(sess, "reject", …)` with one short line; pending ∧ anything else → cleared silently, the turn runs
  as normal (as today's offers).
- **Button**: new WS action `resolve {run_id, decision: approve|reject}` → `submit_turn(resolve_turn, …)`
  (chatbot.py:315), so it serialises with speech on the one TurnController worker (session.py:42-125) and can never
  race a "yes".
- **`resolve_pending(sess, decision, via, run_id=None)`**: run_id given and ≠ the pending one (or nothing pending) →
  `error {scope: turn}` "that request expired", nothing written, `sess.pending` untouched. Else recompute the plan;
  hash mismatch → "that changed since I asked — ask again", nothing written. Approve → the write (`sources.forget`,
  the push) with its own one commit; then `trace.event("decision", run_id, decision, via)`, `card_update {id,
  status, actions: [], commit?}` (`id` = the awaiting card's id; the card also carries `run_id`), one spoken line.
- **Spoken forget** is new (today Forget is button-only): `forget_command(text)` ("forget <title>" / "remove the
  <title> source"), matched against `sources.manifest_entries()` titles; no match or two matches → one spoken line
  ("Which one: A or B?"), no pending set.
- **Push (47b)**: "send that to my reMarkable" proposes `{card_id, title, pages}` for `llm.last_card`; approve calls
  the 47b push; no card → error {scope: turn} "nothing to send". The kind registry is a dict `kind → (plan, apply,
  spoken)` in one new module `agent/approvals.py`, so 47b adds one entry.
- `forget_source` action keeps its name but no longer writes: it proposes. `confirm: true` is gone; the protocol row
  (docs/protocol.md:47) is rewritten, `resolve` and `card_update` rows added (the `card_update` row itself is PRD 51's).
- **Replay**: the awaiting / resolved card is stored as an assistant turn with a marker (as PRD 43 cards,
  chatbot.py:1313-1326) and `card_update` folds into it, so a reconnect shows the current state, never live buttons
  for an expired request.

### Decisions
- One slot, one turn (the `_deeper_offer` shape) over a queue of approvals: the user can hold one question in mind;
  a stale "yes" never applies to something older.
- Resolve through the TurnController over a direct handler: voice and button share one path and one ordering.
- Approval binds the content hash, not just the run_id: a source re-ingested between ask and click is refused.
- Chose a separate `is_approval` (explicit words: yes/delete/forget/send/approve/confirm) over widening
  `_AFFIRMATIVE_WORDS` (llm.py:550-551): writes need explicit words while offers (PRD 45/46) keep `is_affirmative`.
- Chose `save_card` outside approval over gating it: it is an explicit user action that only adds a page; approvals are
  for deletes and writes leaving the vault.
- Chose `via: typed` for a typed "yes", same path as spoken, over folding it into `voice`.
- Chose one short line for a spoken "no"/Reject ("Left it.") and a silent clear for any other turn over always
  acknowledging.
- Chose independence from PRD 51's `card_update` `markdown` (decided there): 52 needs only status, actions, commit.
- Chose a proposal card in the transcript plus a toast linking to it for a Sources-view Forget click over a
  view-local confirm, so the one model holds.
- Depends on PRD 51 (`actions`, `card_update`, run pages; PRD 51's shapes are canonical). Risk: 51 changes shape; mitigation: land 51 first and
  keep this PRD to one `kind` registry plus two handlers.

- Chose a per-session slot over llm.py globals (where `_deeper_offer` lives): approvals are connection state and
  must die with the socket, like hearing/playback (session.py:22-27).

## Data model / state shape
- `Session.pending`: `{run_id: str, kind: "forget"|"push", proposal: dict, hash: str, created: float}` | None.
  Invariant: at most one; cleared by the first turn that starts after it; never persisted.
- Proposal (forget): `{sha, page, raw, cleaned: [page], memory: [entry text]}`; (push): `{card_id, title, pages}`.
- Event: `decision {run_id, kind, decision: approve|reject|expired|changed, via: voice|typed|button}` (no text in prod).
- Card (PRD 51): `status: awaiting|done|rejected`, `actions: [{id, label, kind: approve}, {id, label, kind: reject}]` while awaiting (PRD 51's action shape), `commit?` on `card_update`.

## Test scenarios
1. `tests/session/test_approvals.py`: a "forget the Morrowind guide" turn → an awaiting card with Approve/Reject
   actions, a spoken question, `sess.pending.kind == "forget"`; the source page, raw file and manifest line still
   exist; no new commit.
2. Same file: then a spoken "yes" → page, raw file and manifest line gone, one commit `wiki: forgot …`, a `decision`
   event `{decision: approve, via: voice}`, `card_update {id, status: done, commit}`, one spoken line; `sess.pending` None.
3. Same file: `resolve {run_id, decision: approve}` instead of "yes" → the same files gone and the same messages,
   `via: button`; `resolve {decision: reject}` → nothing written, `card_update {id, status: rejected}`.
4. Same file: "what's the weather" after the proposal → nothing written, no spoken mention, pending None; a spoken
   "no" → nothing written, one short line ("Left it."), `decision {reject, via: voice}`; a typed "yes" resolves as
   approve with `via: typed`; "yeah sure" does not approve a forget; a later `resolve` for that run_id → `error {scope: turn}` "that request expired", nothing written.
5. Same file: proposal A, then proposal B; `resolve` with A's run_id → "that request expired", B still pending and
   intact; the source changed between ask and approve (hash differs) → "changed" error, nothing written.
6. `tests/session/test_approvals.py`: the run page for the run_id shows the proposal body while pending and the
   commit hash/message after approval.
7. `tests/ui/test_approvals_ui.py` (app_fixture.mjs): Approve click disables both buttons until `card_update`; the
   card then shows "done" without buttons; a reconnect replay renders the resolved card, not live buttons; the
   Sources Forget button no longer calls `confirm()` and sends `forget_source {sha}`.
8. `tests/session/test_approvals.py`: "send that to my reMarkable" with a card → awaiting push card, push stub not
   called; "yes" → stub called once with the card; with no card → "nothing to send", no pending.
9. `tests/test_protocol_doc.py` green with the `resolve` row, the rewritten `forget_source` row and `card_update`.

## Out of scope
Multi-step or chained approvals, approvals from another device, timeouts beyond one turn, undo of an approved write,
approvals for memory ops or wiki editor saves.

## Open design questions
None open.
