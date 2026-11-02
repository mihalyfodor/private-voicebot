"""PRD 52 scenarios 1-6 and 8: one pending approval per session, decided by the next turn or a card
button (docs/52-approvals.md). Scenario 7 is tests/server/test_approvals_ui.py, 9 tests/test_protocol_doc.py."""
import os
from unittest.mock import patch

import pytest
from fastapi.testclient import TestClient

from agent import approvals
from server import app
from server import turns as server_turns
from server import ws as m_ws
from server import panels
from server import speech
from agent import conversation
from agent import written
from memory import ops
from ingest import notebooks as ingest_notebooks
from agent import runs
from ingest import pipeline
from vault import manifest
from vault import store
from server.session import Session
from tests.conftest import (MORROWIND, SENTINEL, button_turn, commits, fake_tts, read, seed, session_turn,
                            spoken_lines, trace_lines, trace_text)

pytestmark = pytest.mark.usefixtures("fresh_llm")


def _of(sent: list, kind: str) -> list:
    return [m for m in sent if m["type"] == kind]


def _still_there(m: dict) -> bool:
    return (store.page_exists(f"{m['page']}.md") and os.path.exists(store._path(m["raw"]))
            and any(e["sha256"] == m["sha"] for e in manifest.manifest_entries()))


def _gone(m: dict) -> bool:
    return (not store.page_exists(f"{m['page']}.md") and not os.path.exists(store._path(m["raw"]))
            and not any(e["sha256"] == m["sha"] for e in manifest.manifest_entries()))


def _ask(sess, turn: int = 1, text: str = "forget the Morrowind guide") -> tuple:
    """The proposal turn: ``(sent, card)``."""
    sent = session_turn(sess, turn, text)
    (card,) = _of(sent, "card")
    return sent, card


def _decisions() -> list:
    return [{k: r[k] for k in ("run_id", "workflow", "decision", "via")} for r in trace_lines(kind="decision")]


# --------------------------------------------------------------- 1. the proposal writes nothing


def test_a_forget_turn_proposes_an_awaiting_card_and_writes_nothing(morrowind):
    sess, before = Session(), len(commits())
    sent, card = _ask(sess)
    assert card["status"] == "awaiting" and card["workflow"] == "forget"
    assert [(a["kind"], a["label"]) for a in card["actions"]] == [("approve", "Approve"), ("reject", "Reject"),
                                                                  ("open", "Open run")]
    assert card["run_id"] == sess.pending["run_id"] and card["run_url"] == f"/runs/{card['run_id']}"
    assert morrowind["page"] in card["markdown"] and morrowind["raw"] in card["markdown"]
    assert "projects/falcon" in card["markdown"] and "people: Vivec" in card["markdown"]
    (line,) = spoken_lines(sent)
    assert line == f"That removes {MORROWIND}, its raw file, its notes on 1 page and 1 memory entry. Go ahead?"
    assert sess.pending["kind"] == "forget" and sess.pending["card_id"] == card["id"]
    assert sess.pending["proposal"] == {"sha": morrowind["sha"], "title": MORROWIND, "page": morrowind["page"],
                                        "raw": morrowind["raw"], "images": [], "cleaned": ["projects/falcon"],
                                        "memory": ["people: Vivec"]}
    assert sess.pending["hash"] == approvals.digest(sess.pending["proposal"])
    assert _still_there(morrowind) and len(commits()) == before
    assert ops.source_entries(MORROWIND) == ["people: Vivec"]


def test_forget_plan_is_the_dry_run_of_forget(morrowind):
    plan = pipeline.forget_plan(morrowind["sha"])
    assert _still_there(morrowind)
    result = pipeline.forget(morrowind["sha"])
    assert (result["path"], result["raw"], result["memory"]) == (plan["page"], plan["raw"], len(plan["memory"]))
    assert result["cleaned"] == plan["cleaned"] == ["projects/falcon"]
    with pytest.raises(pipeline.IngestError):
        pipeline.forget_plan(morrowind["sha"])


def test_no_match_or_two_matches_is_one_line_and_nothing_pending(morrowind):
    sess = Session()
    sent = session_turn(sess, 1, "forget the Daggerfall guide")
    assert spoken_lines(sent) == ["I have no source called Daggerfall guide."]
    assert not _of(sent, "card") and sess.pending is None
    with patch.object(pipeline, "list_sources", lambda: [{"sha": "a", "title": "Morrowind guide"},
                                                        {"sha": "b", "title": "Skyrim guide"}]):
        sent = session_turn(sess, 2, "forget the guide")
    assert spoken_lines(sent) == ["Which one: Morrowind guide or Skyrim guide?"] and sess.pending is None


# --------------------------------------------------------------- 2. a spoken "yes" deletes it


def test_a_spoken_yes_forgets_it_in_one_commit(morrowind):
    sess = Session()
    _sent, card = _ask(sess)
    before = len(commits())
    run_id = sess.pending["run_id"]
    sent = session_turn(sess, 2, "yes", origin="spoken")
    assert _gone(morrowind) and ops.source_entries(MORROWIND) == []
    assert f"(source: {morrowind['raw']})" not in read("projects/falcon.md")
    log = commits()
    assert len(log) == before + 1 and log[0].split(" ", 1)[1] == f"wiki: forgot {morrowind['raw']}"
    assert _decisions() == [{"run_id": run_id, "workflow": "forget", "decision": "approve", "via": "voice"}]
    (update,) = _of(sent, "card_update")
    assert update["id"] == card["id"] and update["status"] == "done"
    assert update["commit"] == log[0].split(" ", 1)[0] and update["actions"] == [runs.OPEN]
    assert [a["kind"] for a in update["actions"]] == ["open"]           # no Approve / Reject left
    assert spoken_lines(sent) == ["Forgotten."] and sess.pending is None
    assert _of(sent, "sources") == [{"type": "sources", "sources": []}]   # the Sources view refreshes


# --------------------------------------------------------------- 3. the card's buttons


def test_the_approve_button_forgets_it_the_same_way(morrowind):
    sess = Session()
    _sent, card = _ask(sess)
    before = len(commits())
    sent = button_turn(server_turns.resolve_turn, sess, 2, card["run_id"], "approve")
    assert _gone(morrowind) and len(commits()) == before + 1
    assert _decisions() == [{"run_id": card["run_id"], "workflow": "forget", "decision": "approve", "via": "button"}]
    (update,) = _of(sent, "card_update")
    assert (update["id"], update["status"]) == (card["id"], "done") and update["commit"]
    assert spoken_lines(sent) == ["Forgotten."] and sess.pending is None


def test_the_reject_button_writes_nothing(morrowind):
    sess = Session()
    _sent, card = _ask(sess)
    before = len(commits())
    sent = button_turn(server_turns.resolve_turn, sess, 2, card["run_id"], "reject")
    assert _still_there(morrowind) and len(commits()) == before
    (update,) = _of(sent, "card_update")
    assert (update["id"], update["status"], update["actions"]) == (card["id"], "rejected", [runs.OPEN])
    assert spoken_lines(sent) == [approvals.REJECTED_LINE]
    assert _decisions()[-1]["decision"] == "reject" and _decisions()[-1]["via"] == "button"


def test_the_resolve_action_goes_through_the_turn_worker(morrowind):
    """The WS round trip: `text_turn` proposes, `resolve {run_id, decision}` decides on the worker."""
    panels.greeted = True
    with patch.object(speech, "synthesize", fake_tts), TestClient(app.app).websocket_connect("/ws") as ws:
        ws.send_json({"action": "text_turn", "text": "forget the Morrowind guide"})
        card = _receive(ws, "card")
        _played(ws)
        ws.send_json({"action": "resolve", "run_id": card["run_id"], "decision": "approve"})
        update = _receive(ws, "card_update")
        _played(ws)
        ws.send_json({"action": "resolve", "run_id": card["run_id"], "decision": "maybe"})
        refused = _receive(ws, "error")
    assert update["status"] == "done" and _gone(morrowind)
    assert refused["scope"] == "turn" and "unknown decision" in refused["text"]


def _played(ws) -> None:
    """Answer the spoken line's `speech_end` the way the browser does, then wait for idle."""
    end = _receive(ws, "speech_end")
    ws.send_json({"action": "playback_done", "turn": end.get("turn")})
    _receive(ws, "state", value="idle")


def _receive(ws, kind: str, **match) -> dict:
    while True:
        msg = ws.receive_json()
        if msg["type"] == kind and all(msg.get(k) == v for k, v in match.items()):
            return msg


# --------------------------------------------------------------- 4. other turns, "no", typed "yes", stale clicks


def test_another_turn_clears_it_silently(morrowind):
    sess = Session()
    _sent, card = _ask(sess)
    sent = session_turn(sess, 2, "what's the weather", reply="[neutral] Sunny and mild.")
    assert _still_there(morrowind) and sess.pending is None
    assert spoken_lines(sent) == ["Sunny and mild."] and not _of(sent, "card_update")
    assert _decisions()[-1]["decision"] == "expired"
    late = button_turn(server_turns.resolve_turn, sess, 3, card["run_id"], "approve")
    assert _of(late, "error") == [{"type": "error", "scope": "turn", "text": approvals.EXPIRED}]
    assert _still_there(morrowind)


def test_a_spoken_no_leaves_it_with_one_line(morrowind):
    sess = Session()
    _ask(sess)
    sent = session_turn(sess, 2, "no", origin="spoken")
    assert _still_there(morrowind) and sess.pending is None
    assert spoken_lines(sent) == ["Left it."]
    assert _decisions()[-1]["decision"] == "reject" and _decisions()[-1]["via"] == "voice"


def test_a_typed_yes_approves_with_via_typed(morrowind):
    sess = Session()
    _ask(sess)
    session_turn(sess, 2, "yes, delete it")
    assert _gone(morrowind) and _decisions()[-1]["via"] == "typed" and _decisions()[-1]["decision"] == "approve"


def test_yeah_sure_does_not_approve_a_forget(morrowind):
    sess = Session()
    _ask(sess)
    session_turn(sess, 2, "yeah sure")
    assert _still_there(morrowind) and sess.pending is None and _decisions()[-1]["decision"] == "expired"


FORGET_PENDING = {"kind": "forget", "proposal": {"title": MORROWIND}}
PUSH_PENDING = {"kind": "push", "proposal": {"title": "Plan"}}


@pytest.mark.parametrize("text, pending, approve, reject", [
    # kind-independent yes
    ("yes", FORGET_PENDING, True, False), ("Yes please.", FORGET_PENDING, True, False),
    ("yes please", PUSH_PENDING, True, False), ("yes, go ahead", FORGET_PENDING, True, False),
    ("go ahead", PUSH_PENDING, True, False), ("do it", FORGET_PENDING, True, False),
    ("confirm", FORGET_PENDING, True, False), ("approve", PUSH_PENDING, True, False),
    ("yes, delete it", FORGET_PENDING, True, False), ("yes, delete it", PUSH_PENDING, False, False),
    # a kind's own verb only when it names the proposal
    ("delete it", FORGET_PENDING, True, False), ("delete it", PUSH_PENDING, False, False),
    ("remove the source", FORGET_PENDING, True, False), ("forget the Morrowind guide", FORGET_PENDING, True, False),
    ("delete the Morrowind guide source", FORGET_PENDING, True, False),
    ("delete the Skyrim guide", FORGET_PENDING, False, False),
    ("send it", FORGET_PENDING, False, False), ("send it", PUSH_PENDING, True, False),
    ("send the card", PUSH_PENDING, True, False), ("push it to my reMarkable", PUSH_PENDING, True, False),
    ("forget", FORGET_PENDING, False, False), ("delete", FORGET_PENDING, False, False),
    ("send", PUSH_PENDING, False, False),
    # never a yes
    ("yeah sure", FORGET_PENDING, False, False), ("ok", FORGET_PENDING, False, False),
    ("sure, do it", FORGET_PENDING, False, False), ("yes, but what does it remove", FORGET_PENDING, False, False),
    ("forget the Skyrim guide", FORGET_PENDING, False, False),
    # "forget it/that/this/about…" and "never mind" are a no, whatever follows
    ("forget it", FORGET_PENDING, False, True), ("forget it please", FORGET_PENDING, False, True),
    ("forget it, thanks", FORGET_PENDING, False, True), ("forget that please", FORGET_PENDING, False, True),
    ("forget this", FORGET_PENDING, False, True), ("forget about it", FORGET_PENDING, False, True),
    ("forget about the whole thing", FORGET_PENDING, False, True), ("never mind", FORGET_PENDING, False, True),
    ("never mind that, what time is it", FORGET_PENDING, False, True),
    ("no", FORGET_PENDING, False, True), ("no thanks", PUSH_PENDING, False, True), ("nope", FORGET_PENDING, False, True),
    ("don't", FORGET_PENDING, False, True), ("cancel", FORGET_PENDING, False, True), ("leave it", FORGET_PENDING, False, True),
    ("no, tell me about Vivec instead", FORGET_PENDING, False, False),
])
def test_the_approval_and_rejection_words(text, pending, approve, reject):
    assert (approvals.is_approval(text, pending), approvals.is_rejection(text)) == (approve, reject)


@pytest.mark.parametrize("text", ["forget it please", "forget that please", "forget this", "forget"])
def test_forget_it_never_deletes_the_source(morrowind, text):
    sess = Session()
    _ask(sess)
    session_turn(sess, 2, text)
    assert _still_there(morrowind) and sess.pending is None
    assert _decisions()[-1]["decision"] in ("reject", "expired")


# --------------------------------------------------------------- 5. one slot, bound to its hash


def test_a_newer_proposal_replaces_the_older_one(morrowind):
    sess = Session()
    _sent, first = _ask(sess)
    seed(people=[{"name": "Vivec", "source": MORROWIND}, {"name": "Almalexia", "source": MORROWIND}])
    (second,) = _of(button_turn(server_turns.propose_turn, sess, 2, "forget", morrowind["sha"]), "card")
    assert first["run_id"] != second["run_id"] and sess.pending["run_id"] == second["run_id"]
    held = dict(sess.pending)
    sent = button_turn(server_turns.resolve_turn, sess, 3, first["run_id"], "approve")
    assert _of(sent, "error")[0]["text"] == approvals.EXPIRED
    assert [(u["id"], u["status"]) for u in _of(sent, "card_update")] == [(first["id"], "expired")]   # only A's card
    assert sess.pending == held and _still_there(morrowind)


def test_a_source_that_changed_since_the_ask_is_refused(morrowind):
    sess = Session()
    _sent, card = _ask(sess)
    before = len(commits())
    seed(people=[{"name": "Vivec", "source": MORROWIND}, {"name": "Almalexia", "source": MORROWIND}])
    sent = button_turn(server_turns.resolve_turn, sess, 2, card["run_id"], "approve")
    assert _of(sent, "error") == [{"type": "error", "scope": "turn", "text": approvals.CHANGED}]
    assert _still_there(morrowind) and len(commits()) == before and sess.pending is None
    assert _decisions()[-1]["decision"] == "changed"
    (update,) = _of(sent, "card_update")
    assert update["status"] == "failed" and update["actions"] == [runs.OPEN]


# --------------------------------------------------------------- 6. the run page


def test_the_run_page_shows_the_proposal_then_the_commit(morrowind):
    sess = Session()
    _sent, card = _ask(sess)
    client = TestClient(app.app)
    pending = client.get(card["run_url"])
    assert pending.status_code == 200
    assert "Proposal" in pending.text and morrowind["raw"] in pending.text and morrowind["page"] in pending.text
    assert "projects/falcon" in pending.text and "1 memory entry" in pending.text
    assert "run-awaiting" in pending.text and "Commit" not in pending.text
    session_turn(sess, 2, "yes")
    sha, subject = commits()[0].split(" ", 1)
    done = client.get(card["run_url"]).text
    assert "run-done" in done and "<h2>Commit</h2>" in done
    assert f'<code class="run-sha">{sha}</code> {subject}' in done


def test_an_approval_writes_no_text_into_prod_events(morrowind, prod_profile):
    seed(people=[{"name": f"Vivec {SENTINEL}", "source": MORROWIND}])
    sess = Session()
    _sent, card = _ask(sess, text=f"forget the {MORROWIND}")
    assert SENTINEL in card["markdown"]                      # the card shows it ...
    session_turn(sess, 2, "yes")
    assert SENTINEL not in trace_text()                      # ... the events never do
    (decision,) = trace_lines(kind="decision")
    assert set(decision) >= {"run_id", "workflow", "decision", "via"} and decision["decision"] == "approve"
    page = TestClient(app.app).get(card["run_url"]).text
    assert SENTINEL not in page


# --------------------------------------------------------------- replay


def test_a_replay_folds_the_status_and_never_shows_live_buttons_for_an_expired_request(morrowind):
    sess = Session()
    _sent, card = _ask(sess)
    live = [m for m in m_ws.replay_messages(sess) if m["type"] == "card"]
    assert [a["kind"] for a in live[0]["actions"]] == ["approve", "reject", "open"]
    stale = [m for m in m_ws.replay_messages(Session()) if m["type"] == "card"]   # a reconnect: a new session
    assert stale[0]["status"] == "awaiting" and [a["kind"] for a in stale[0]["actions"]] == ["open"]
    session_turn(sess, 2, "no")
    (done,) = [m for m in m_ws.replay_messages(sess) if m["type"] == "card"]
    assert done["status"] == "rejected" and [a["kind"] for a in done["actions"]] == ["open"]
    roles = [m.get("role") or m["type"] for m in m_ws.replay_messages(sess)]
    assert roles == ["user", "card", "assistant", "user", "assistant"]


# --------------------------------------------------------------- the Sources view's Forget


def test_forget_source_proposes_instead_of_deleting(morrowind):
    sess = Session()
    sent = button_turn(server_turns.propose_turn, sess, 1, "forget", morrowind["sha"])
    (card,) = _of(sent, "card")
    assert card["status"] == "awaiting" and sess.pending["run_id"] == card["run_id"]
    assert _still_there(morrowind)


# --------------------------------------------------------------- 8. push (PRD 47b's entry)


def _with_card() -> dict:
    card = {"id": written.next_card_id(), "markdown": "# Plan\n\n" + "line\n" * 70, "sources": [], "title": "Plan"}
    conversation._cards[card["id"]] = card
    conversation.last_card = card
    return card


def test_send_that_to_my_remarkable_proposes_then_pushes_once(vault):
    card = _with_card()
    sess = Session()
    with patch.object(ingest_notebooks, "push_available", return_value=True), \
            patch.object(ingest_notebooks, "push", return_value={"ok": True, "pages": 2}) as push:
        sent = session_turn(sess, 1, "send that to my reMarkable")
        (proposal,) = _of(sent, "card")
        assert proposal["status"] == "awaiting" and proposal["workflow"] == "push"
        assert sess.pending["proposal"] == {"card_id": card["id"], "title": "Plan", "pages": 2}
        assert spoken_lines(sent) == ["Send the card Plan (about 2 pages) to your reMarkable?"]
        push.assert_not_called()
        sent = session_turn(sess, 2, "yes")
    push.assert_called_once_with(card)
    assert spoken_lines(sent) == ["Sent to your reMarkable."]
    (update,) = _of(sent, "card_update")
    assert update["status"] == "done" and "commit" not in update


def test_send_without_a_card_is_nothing_to_send(vault):
    sess = Session()
    with patch.object(ingest_notebooks, "push_available", return_value=True), patch.object(ingest_notebooks, "push") as push:
        sent = session_turn(sess, 1, "send that to my reMarkable")
    assert _of(sent, "error") == [{"type": "error", "scope": "turn", "text": "nothing to send"}]
    assert sess.pending is None and not _of(sent, "card")
    push.assert_not_called()


# --------------------------------------------------------------- fix round 1 (review findings)


def test_a_notebook_proposal_names_its_page_images_and_binds_them(vault):
    """MAJOR 1: the page PNGs `forget` removes are in the proposal (count + hash)."""
    from tests.conftest import FakeClient, notebook_pages, png_bytes
    from core import llm as core_llm
    import json as _json
    summary = _json.dumps({"title": "Standup", "summary": "Notes.", "entities": [], "unreadable": False})
    with patch.object(core_llm, "_client", FakeClient("Page text one.", "Page text two.", "Page text three.",
                                                      summary, "[]")):
        filed = ingest_notebooks.ingest_notebook(notebook_pages(3), title="Standup")
    sha, stem = filed["sha256"], filed["raw"][:-len(".md")]
    plan = pipeline.forget_plan(sha)
    assert plan["images"] == [f"{stem}/page-0{i}.png" for i in (1, 2, 3)]
    assert "3 page images" in approvals.FORGET.body(plan)
    with open(store._path(f"{stem}/page-04.png"), "wb") as f:      # a page file added after the ask
        f.write(png_bytes(9, 4))
    assert approvals.digest(pipeline.forget_plan(sha)) != approvals.digest(plan)
    os.remove(store._path(f"{stem}/page-04.png"))
    sess = Session()
    sent = button_turn(server_turns.propose_turn, sess, 1, "forget", sha)
    assert _of(sent, "card")[0]["status"] == "awaiting"
    files_before = _vault_files()
    button_turn(server_turns.resolve_turn, sess, 2, sess.pending["run_id"], "approve")
    gone = {f for f in files_before - _vault_files() if not f.startswith("raw/.index/")}   # minus its chunk cache
    assert gone == {filed["raw"], f"{filed['path']}.md", *plan["images"]}      # the images went, nothing else
    assert not os.path.exists(store._path(stem))


def _vault_files() -> set:
    root = store.WIKI_PATH
    return {os.path.relpath(os.path.join(d, f), root) for d, subs, fs in os.walk(root)
            for f in fs if ".git" not in d.split(os.sep) and not d.startswith(os.path.join(root, ".git"))}


def test_a_reconnect_expires_the_old_sessions_pending(morrowind):
    """MAJOR 2: the evicted session's approval ends `expired` (via reconnect); the replay has no buttons."""
    panels.greeted = True
    with patch.object(speech, "synthesize", fake_tts):
        client = TestClient(app.app)
        with client.websocket_connect("/ws") as ws:
            ws.send_json({"action": "text_turn", "text": "forget the Morrowind guide"})
            card = _receive(ws, "card")
            _played(ws)
            with client.websocket_connect("/ws") as ws2:
                replayed = _receive(ws2, "card")
                _receive(ws2, "state", value="idle")
    assert _decisions() == [{"run_id": card["run_id"], "workflow": "forget", "decision": "expired",
                             "via": "reconnect"}]
    assert [a["kind"] for a in replayed["actions"]] == ["open"]
    page = TestClient(app.app).get(card["run_url"]).text
    assert "run-expired" in page and _still_there(morrowind)


@pytest.mark.parametrize("asr_text, asr_raises", [("", False), (None, True)])
def test_an_empty_or_failed_utterance_clears_the_pending(morrowind, asr_text, asr_raises):
    import numpy as np
    from tests.conftest import fake_asr, run_turn
    sess = Session()
    _ask(sess)
    asr = fake_asr(asr_text or "")
    if asr_raises:
        def boom(audio, prompt=None):
            raise RuntimeError("asr down")
        asr.transcribe = boom
    with patch.object(server_turns, "asr", asr):
        run_turn(server_turns.handle_utterance, sess, 2, np.zeros(10, dtype=np.float32))
    assert sess.pending is None and _decisions()[-1]["decision"] == "expired" and _still_there(morrowind)


def test_push_is_not_offered_without_the_service(vault):
    """MAJOR 3: without REMARKABLE_PUSH_URL the command is one spoken line, no pending, no run."""
    _with_card()
    sess = Session()
    sent = session_turn(sess, 1, "send that to my reMarkable")
    assert spoken_lines(sent) == ["The reMarkable push service is not configured."]
    assert sess.pending is None and not _of(sent, "card") and trace_lines(kind="run") == []


def test_a_write_that_fails_still_writes_its_decision_and_run_end(vault):
    """MAJOR 3: the approved push raises → `decision approve` + `run end failed` before the error."""
    _with_card()
    sess = Session()
    with patch.object(ingest_notebooks, "push_available", return_value=True), \
            patch.object(ingest_notebooks, "push", side_effect=RuntimeError("tablet gone")):
        session_turn(sess, 1, "send that to my reMarkable")
        run_id = sess.pending["run_id"]
        sent = session_turn(sess, 2, "yes")
    assert _decisions() == [{"run_id": run_id, "workflow": "push", "decision": "approve", "via": "typed"}]
    ends = [r for r in trace_lines(kind="run") if r["phase"] == "end"]
    assert [(r["run_id"], r["status"], r["reason"]) for r in ends] == [(run_id, "failed", "failed-RuntimeError")]
    (update,) = _of(sent, "card_update")
    assert update["status"] == "failed" and _of(sent, "error")


def test_a_change_between_the_check_and_the_write_is_refused_under_the_lock(morrowind):
    """Backlog 1: `forget` re-checks the plan under the vault lock."""
    sess = Session()
    _sent, card = _ask(sess)
    real = pipeline.forget_plan
    calls = []

    def plan_then_change(sha):
        plan = real(sha)
        calls.append(1)
        if len(calls) == 1:          # the resolve's re-plan passes, then the profile changes
            seed(people=[{"name": "Vivec", "source": MORROWIND}, {"name": "Almalexia", "source": MORROWIND}])
        return plan
    before = len(commits())
    with patch.object(pipeline, "forget_plan", plan_then_change):
        sent = button_turn(server_turns.resolve_turn, sess, 2, card["run_id"], "approve")
    assert _of(sent, "error") == [{"type": "error", "scope": "turn", "text": approvals.CHANGED}]
    assert _still_there(morrowind) and len(commits()) == before
    assert _decisions()[-1]["decision"] == "changed"


@pytest.mark.parametrize("answer", ["yes", "no"])
def test_an_answer_to_the_approval_takes_the_deeper_offer_too(morrowind, answer):
    """Backlog 2: the resolving turn consumes a pending offer like any other turn."""
    sess = Session()
    _ask(sess)
    conversation._deeper_offer = {"question": "Shall I dig deeper?"}
    session_turn(sess, 2, answer)
    assert conversation._deeper_offer is None


def test_a_click_on_a_stale_card_updates_that_card(morrowind):
    """Backlog 3: the late click also gets `card_update` expired (Open run only) for its card."""
    sess = Session()
    _sent, card = _ask(sess)
    session_turn(sess, 2, "what's the weather", reply="[neutral] Sunny.")
    sent = button_turn(server_turns.resolve_turn, sess, 3, card["run_id"], "approve")
    assert _of(sent, "card_update") == [{"type": "card_update", "id": card["id"], "status": "expired",
                                         "steps": _of(sent, "card_update")[0]["steps"], "actions": [runs.OPEN]}]
    assert _of(sent, "error")[0]["text"] == approvals.EXPIRED
    (replayed,) = [m for m in m_ws.replay_messages(sess) if m["type"] == "card"]
    assert replayed["status"] == "expired"


def test_a_failed_proposal_expires_the_older_one_first(morrowind):
    """Backlog 4: the older pending is expired before the new plan; a failed plan leaves nothing pending."""
    sess = Session()
    _ask(sess)
    sent = button_turn(server_turns.propose_turn, sess, 2, "forget", "nope")
    assert _of(sent, "error") and sess.pending is None
    assert _decisions()[-1]["decision"] == "expired"


def test_the_awaiting_step_says_to_clean_and_the_applied_step_cleaned(morrowind):
    """Backlog 7."""
    sess = Session()
    _sent, card = _ask(sess)
    assert card["steps"][0]["detail"] == "1 page to clean, 1 memory entry"
    sent = session_turn(sess, 2, "yes")
    (update,) = _of(sent, "card_update")
    assert update["steps"][-1]["detail"].startswith("1 page cleaned, 1 memory entry removed · commit ")


def test_the_palette_new_topic_clears_the_pending(morrowind):
    """Backlog 9: the PRD's "the first turn after it" includes the palette's New topic."""
    from tests.conftest import FakeClient
    from core import llm as core_llm
    sess = Session()
    _ask(sess)
    with patch.object(core_llm, "_client", FakeClient("A short summary.")):
        button_turn(server_turns.new_topic_turn, sess, 2)
    assert sess.pending is None and _decisions()[-1]["decision"] == "expired"


# --------------------------------------------------------------- refactor 52: slot lock, sweep, rows


def _notebook(vault, pages: int = 2) -> dict:
    from tests.conftest import FakeClient, notebook_pages
    from core import llm as core_llm
    import json as _json
    summary = _json.dumps({"title": "Standup", "summary": "Notes.", "entities": [], "unreadable": False})
    texts = [f"Page text {i}." for i in range(pages)]
    with patch.object(core_llm, "_client", FakeClient(*texts, summary, "[]")):
        return ingest_notebooks.ingest_notebook(notebook_pages(pages), title="Standup")


def test_a_stray_file_in_a_notebooks_page_folder_is_not_swept(vault):
    filed = _notebook(vault)
    sha, stem = filed["sha256"], filed["raw"][:-len(".md")]
    clean = pipeline.forget_plan(sha)
    stray = store._path(f"{stem}/.DS_Store")
    with open(stray, "wb") as f:
        f.write(b"junk")
    plan = pipeline.forget_plan(sha)
    assert plan["images"] == clean["images"] and not any("DS_Store" in i for i in plan["images"])
    assert approvals.digest(plan) == approvals.digest(clean)
    sess = Session()
    button_turn(server_turns.propose_turn, sess, 1, "forget", sha)
    button_turn(server_turns.resolve_turn, sess, 2, sess.pending["run_id"], "approve")
    assert os.path.exists(stray) and not os.path.exists(store._path(f"{stem}/page-01.png"))


def test_the_run_page_lists_the_page_images_the_card_counts(vault):
    filed = _notebook(vault)
    sess = Session()
    sent = button_turn(server_turns.propose_turn, sess, 1, "forget", filed["sha256"])
    card = _of(sent, "card")[0]
    stem = filed["raw"][:-len(".md")]
    html = TestClient(app.app).get(card["run_url"]).text
    assert "2 page images" in card["markdown"]
    assert f"{stem}/page-01.png" in html and f"{stem}/page-02.png" in html


def test_a_disconnect_expire_racing_a_resolve_yields_one_decision(morrowind):
    import threading
    gate = threading.Event()
    resolver = {}

    class Racy(Session):
        """Its first `pending` read on the resolving thread parks, so an expire can try to slip in."""
        @property
        def pending(self):
            value = self.__dict__.get("_p")
            if value is not None and threading.current_thread() is resolver.get("t") and not gate.is_set():
                gate.set()
                threading.Event().wait(0.3)
            return value

        @pending.setter
        def pending(self, value):
            self.__dict__["_p"] = value

    sess = Racy()
    _ask(sess)
    out = {}

    def resolve():
        out["o"] = approvals.resolve_pending(sess, "approve", "button")

    resolver["t"] = t = threading.Thread(target=resolve)
    t.start()
    assert gate.wait(5)
    approvals.expire(sess, "disconnect")
    t.join(10)
    assert len(_decisions()) == 1
