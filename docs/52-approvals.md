Status: refactoring

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
  status, actions: [open], commit?}` (`id` = the awaiting card's id; the card also carries `run_id`), one spoken line.
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
- Implementation: the module is a root `approvals.py` (the `agent/` layer does not exist yet); it moves to
  `agent/approvals.py` at PRD 50 step 9. `is_approval` / `is_rejection` live there, not in llm.py.
- Implementation: `KINDS[kind]` is a NamedTuple whose first three fields are `plan` / `apply` / `spoken`; it also
  carries the card title/body, the run page's Proposal rows, a count line and the done line.
- Implementation: approval words: the first word one of yes/approve(d)/confirm(ed)/delete/forget/send, the rest from
  a small filler list (please, it, go, ahead, do, thanks, ok…), ≤ 6 words. Rejection: no/nope/nah/cancel/reject/
  don't/stop/leave/keep/never mind, ≤ 4 words; "forget it" / "forget about it" / "forget that" are a no, never an
  approval of a forget.
- Implementation: the proposal adds `title` (`{sha, title, page, raw, cleaned, memory}`); `cleaned` holds page names
  without `.md`. `Session.pending` also keeps `card_id` and the card's `steps`, so the resolve updates that card.
- Implementation: the `decision` event carries the approval kind as `workflow` (an event's `kind` is its envelope's
  and always wins); `decision` and `via` are fixed vocabularies, `proposal` rows keep only `path`/`title`.
- Implementation: run events: the proposal turn writes `run start` and a `propose` step (rows of paths/titles, the
  memory entries only as a count); the run stays open while pending (the page derives `awaiting`); the decision
  closes it under the proposing run_id: done/ok, rejected/rejected, expired/expired, failed/changed. An approval's
  `apply` step carries the commit and the removed paths. The run page gains a Proposal and a Commit section.
- Implementation: a resolved card keeps "Open run" (`actions: [open]`), so its run page stays one click away;
  Approve / Reject are what `card_update` drops.
- Implementation: a silent clear writes `decision expired` (via the clearing turn's origin) and leaves the old
  card's buttons on the page; a click on them is "that request expired". A replay drops them (a reconnect is a new
  session, so nothing is pending). The palette's New topic clears a pending approval too.
- Implementation: a changed proposal also turns its card `failed` (Open run only) besides the `error`; a write
  that raises turns it `failed` with the reason sentence.
- Implementation: the spoken forget matches the normalised title (a leading "the" dropped) exactly first, else the
  titles containing every word; none → "I have no source called <title>." The regex never takes "forget it",
  "forget about/to/what…", "don't forget…".
- Implementation: "send that to my reMarkable" targets only `llm.last_card` (the open-page target is 47b's);
  `notebooks.push(card)` raises NotImplementedError until 47b lands (tests patch it); the page estimate is
  lines / 60, at least 1.
- Implementation: `forget_source` with an unknown sha is answered at once (`error` sources); a known one queues
  `propose_turn`. The toast reuses the intake toast element; clicking it opens Chat at the card.
- Implementation: the proposal and its answer are recorded in the conversation (`llm.record_user` + the canned
  lines), so a reconnect replays the request, the card and the question in order.
- Review fix (supersedes the approval-word bullet above): "forget it/that/this/about…" and "never mind" are a no
  whatever follows. A kind's own verbs approve only that kind and only when they name the proposal ("it", "the
  source" / "the card", or its title): delete/remove/forget for a forget, send/push for a push; a bare "forget" or
  "delete" is neither. yes/approve(d)/confirm(ed)/go ahead/do it stay kind-independent ("yes, delete it" approves a
  forget, not a push). `is_approval` takes the pending approval for this.
- Review fix: a notebook's page images are in the forget proposal (`images`, "3 page images" on the card and in the
  question) and in its hash — the files in the notebook's page folder as they are now, which is what `forget`
  removes.
- Review fix: `sources.forget(sha, expected=plan)` re-checks the approved plan under the vault lock; a difference is
  `IngestError(409)` → the same "changed" outcome, nothing written.
- Review fix: a pending approval also ends (`decision expired`) when its socket is replaced (via `reconnect`) or
  closes (via `disconnect`), and on an empty or failed utterance (via `voice`); an expired card is folded to status
  `expired` (a card status 52 adds, with its chip) so a replay shows it settled.
- Review fix (amends the silent-clear bullet): a click on a stale card gets, besides the error, `card_update`
  for that run's card (status `expired`, or its own resolved status; Open run only).
- Review fix: until PRD 47b `notebooks.push_available()` is False and "send that to my reMarkable" is one spoken
  line, "Sending to the reMarkable isn't available yet.", with no pending and no run. A write that raises after
  approval still writes `decision approve` and `run end failed` before the error.
- Review fix: the proposal turn expires the older pending before planning (a failed plan leaves nothing pending);
  the answering turn also drops an open deeper / write-up offer, as any turn does.
- Review fix: "forget <title>" needs a source-kind word at the end (source, guide, notebook, file, book, document,
  pdf, email, article, note(s)) or the "the <title> source" form, so "forget my address" or "forget Anna" reach
  the model; the kind word is dropped when no title matches with it.
- Review fix: the awaiting card's step reads "1 page to clean, 1 memory entry"; the applied step "1 page cleaned, 1
  memory entry removed · commit <sha>".
- Note: the registry's `writes` flag stays metadata only (as PRD 50 says); approvals are routed by the `forget` /
  `push` tools themselves, not by the flag.
- Fact fix: the Technical approach's `card_update` now reads `actions: [open]` (it said `actions: []`, which
  contradicted the "keeps Open run" bullet above).

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
