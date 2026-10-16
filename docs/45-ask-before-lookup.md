Status: exploration

# 45 — Ask before lookup (recall calibration)

## Overview
Auto-recall quotes filed text into every turn whose words happen to match: a mis-transcribed "Are you listening,
Uncle?" pulled a 6,073-char Morrowind chunk into a four-word chat turn and padded the spoken reply to 342 chars (92
typed). The search is cheap (0.15 s, ~0.02 s prefill); the cost is the longer reply and its TTS. This PRD makes a
spoken turn ask before it looks up whenever the evidence is weak: strong evidence answers grounded as today, weak
evidence gets one short offer ("Want me to check your Morrowind guide?"), and a "yes" then gives the grounded answer.

## User stories
- As a user talking, chit-chat that happens to contain a name from a guide ("Are you listening, Uncle?") gets a
  normal short reply, at most with a one-line offer to check the guide, never a walkthrough paraphrase.
- As a user asking a real question aloud ("What does Caius want?", "I'm at Covenant and the turrets shoot me"), I get
  the grounded short answer and the deeper offer, as today.
- As a user, I answer the offer with "yes" and get the grounded answer to my earlier question; "no" or anything else
  drops it silently and the conversation goes on.
- As a user typing, nothing changes: typed content questions still answer inline and long (PRD 43).

## UI/UX
- No new UI. The medium band adds one fixed spoken sentence at the end of a short reply:
  "Want me to check your <title>?" (the source title up to the first ":" or " (").
- The offer is never made twice in a row; a second weak turn right after an offer is answered without recall or offer.
- After "yes" the turn shows the usual `dig_deeper`-style filler and the grounded answer, ending with the deeper offer.

## Technical approach
- **Score facts** (measured on a copy of the dev vault, the three guides + EPUBs): `score` is BM25 divided by the
  number of topic words (`wiki.py:1197`), so one rare proper term scores *high*: "Uncle" 18.1, "Who is Vivec?" 23.5,
  "What does Caius want?" 16.6, against 3.0–12.8 for the battery's multi-word content questions (A8 3.4, A11 3.0,
  A1 11.1, A5 12.8). A score-only ladder cannot put Uncle below A1, so the band uses score *and* the kind of evidence.
- **Band** — new `recall_band(user_text, page, hits) -> "high" | "medium" | "low"` in `llm.py`, next to
  `recall_for` (`llm.py:762`), computed from what `_page_recall` (`llm.py:801`) and `_recall_hits` (`llm.py:781`,
  floor `wiki.recall_min_score()` `wiki.py:1236`, default `RECALL_MIN_SCORE = 4.5` `wiki.py:913`) already return:
  - low: no page and no kept hit (today's rules: `RECALL_MIN_WORDS = 4` `llm.py:432`, `COMMAND_WORDS`, the PRD 34/42
    precision rule), or the gate fails: neither `content_question()` (`llm.py:673`) passes nor a
    `wiki.proper_match` (`wiki.py:1203`) term is in the top hit.
  - high: a named page or a named source (`hit["named"]`); or `content_question()` and the top hit has a proper match
    or a score ≥ `RECALL_HIGH_SCORE`; or, without a content question, a proper bigram held whole (`hit["phrases"]`,
    "Caius Cosades") or ≥ 2 proper terms matched ("Thieves Guild in Riften").
  - medium: everything else with a kept hit — a non-content turn carried by one proper term (Uncle, Vivec,
    Balmora), a content question matched on plain words only below `RECALL_HIGH_SCORE` ("2024 Paris Olympics" →
    an architecture book on "open", "paris", 6.5).
  - `RECALL_HIGH_SCORE = 8.0` sits in `wiki.py` next to `RECALL_MIN_SCORE`; initial value from the one plain-word,
    unnamed battery hit (B4 6.5) against the named/proper ones; to be confirmed in Impl against the band table
    (scenario 1). No `recall_high_score` setting.
- **Spoken only.** `ask_events` (`llm.py:867`) applies the band when `origin != "typed"`; typed turns keep
  `recall_for` exactly as today. Focus (`focus_block`), deeper and Wikipedia "yes" turns skip the band.
- **Medium turn:** no recall text in the message; the per-turn note `RECALL_ASK_NOTE` says "answer briefly as you
  normally would, do not end with a question"; the server appends `RECALL_OFFER.format(title=…)` as the last delta
  (deterministic, the same seam the model-ended offers use). The held-reply fallback (`holding`, `llm.py:892`) is
  off for a medium turn: the offer replaces it.
- **Pending offer** reuses `_deeper_offer` (`llm.py:293`): `{"recall": question, "title": t}`. The existing
  one-turn pop (`llm.py:879`) gives "no"/anything else = cleared silently. `is_affirmative` (`llm.py:556`) → the turn
  re-runs `recall_for(question)` (forced high, `_turn_sources` recorded), the message is "yes" + the block + a note
  naming the earlier question, and the normal short grounded rule applies, deeper offer included. `_last_context`
  (`llm.py:1010`) stores the earlier question, so "write it up" (`wants_writeup`) works on it.
- `_log` (`llm.py:1130`) appends ` | recall: <band>` for spoken turns, for tuning from `session.log`.

### Decisions
- Band on score + evidence kind over a score ladder alone: the score is per topic word, so a lone rare name always
  outranks a real question (measured above).
- Spoken only over both origins: TTS is the cost; typed long answers are PRD 43's deliberate policy, and the typed
  battery then guards against regressions.
- `CONTENT_WORDS` unchanged (no who/where/which): adding them would also change PRD 43 typed length; "Who is Vivec?"
  aloud gets one offer, which is the intended ask-first.
- Server-appended offer over prompting the model to end with it: PRD 34 needed per-turn repetition for 6/6, and
  `ends_with_offer` detection is not needed when the server writes the sentence.
- Risk: offers on chat that mentions a corpus name feel naggy; mitigation: one sentence, never twice in a row.
- Risk: real spoken questions land in medium and need a "yes"; mitigation: the band table and the band in
  `session.log`; tune `RECALL_HIGH_SCORE`, not the rules.

## Data model / state
- `_deeper_offer` gains a third shape `{"recall": str, "title": str}`; still lives exactly one turn.
- Constants: `RECALL_HIGH_SCORE` (`wiki.py`), `RECALL_OFFER`, `RECALL_ASK_NOTE` (`llm.py`). No settings, no
  protocol change.

## Test scenarios
Unit tests use `FakeClient` / `StreamClient` from `tests/conftest.py` and the `fallout` / `both` FAQ fixtures; new file
`tests/llm/test_ask_before_lookup.py`.
1. Band table (marker `live`, skips without `~/llm-wiki-dev`; deterministic, no LLM): over a copy of the dev vault,
   "Are you listening, Uncle?" → medium; "Who is Vivec?" → medium; "What does Caius want?" → high; "Where is Caius
   Cosades?" → high; every A/C battery question in `tests/qa/stress.py` that recalls today → high; "How long should
   I steep green tea?" → low. Printed with score, matched and proper terms.
2. Spoken four-word mis-transcription against a fixture where a proper term carries one hit: the message has no
   `<recall>` block, the streamed reply ends with "Want me to check your <title>?", `_deeper_offer["recall"]` is
   the question, one completion only, no fallback retry.
3. Spoken genuine four-word content question with a proper hit: `<recall>` block present, offer from
   `DEEPER_OFFERS` as today, `_deeper_offer` holds source and parts.
4. Typed: the same mis-transcription and the same content question → messages identical to today's `recall_for`
   output (block present), no recall offer; a typed content question still runs at the `long` budget.
5. "yes" after the offer → the message carries the `<recall>` block for the earlier question, the reply ends with a
   deeper offer, `_last_context["question"]` is the earlier question, `_deeper_offer` holds the source.
6. "no" after the offer → no block, no offer appended, `_deeper_offer` None; "no, what's the weather" → the weather
   turn runs normally and injects nothing.
7. Two weak spoken turns in a row → the first ends with the offer, the second has neither block nor offer.
8. Gate: a spoken non-content turn whose hits are two plain words ("tell me something nice about dragons" style,
   fixture-matched) → low: no block, no offer.
9. `session.log` spoken lines end with `| recall: high|medium|low`; `tests/session/test_text_input.py` stays green.
10. Battery: re-run `tests/qa/stress.py` (typed): ≥ 37 correct of 43 scored (PRD 43 results: 37 / 3 / 2, 2 n/a of
    45); manual spoken pass of scenarios 2, 3, 5, 6 in the app, reply length of the Uncle turn ≤ 120 chars.

## Out of scope
- Embeddings or a re-ranker; hedged wording ("I think…"); changing `RECALL_MIN_SCORE` or the PRD 34/42 precision
  rules.
- Any change to typed turns, including PRD 43 long answers; Wikipedia routing (PRD 37).
- Remembering declined offers beyond the next turn; a settings toggle for the band.
- `eval/` re-tuning: it holds the memory/ingest scripts (`eval/scripts/*.yaml`), no recall cases.
