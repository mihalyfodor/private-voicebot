Status: done

# 34 — Grounded answers: look it up first, short reply, offer to dig deeper

## Overview
PRD 32 made filed documents reachable, but the model still prefers its own knowledge, and answers are one-size.
This PRD makes "look it up" the default whenever a source or page may cover the question, keeps the first answer
short with an offer to dig deeper, and lets a "yes" unlock a longer, source-read answer for that turn. It also
tightens auto-recall so one stray keyword ("cook", "today") no longer drags a game guide into dinner plans.

## User stories
- As a user, when I ask about something I filed, Wanko checks the source before answering, and says where it came from.
- As a user, I first get a 1–3 sentence answer plus "want me to dig deeper?"; saying yes gets a fuller answer that
  reads the surrounding text, without me changing the reply-length setting.
- As a user, small talk never gets a guide snippet stapled to it.

## Technical approach
- **Prompt rules** (system prompt, character-independent block, present only when the recall tools are registered —
  the eval harness runs tool-free): if the wiki index or a recall block names a source on the topic, call
  `search_wiki`/`read_source` (or use the recall) before answering from memory; name the source title once in a short
  natural clause inside a sentence (varied, never a separate closing sentence); whenever the answer comes from a
  document passage, keep it to 1–3 sentences and make the last sentence exactly "Want me to dig deeper?" (a document
  always has more around a passage). Off-topic recall is to be ignored silently. The per-turn `<recall>` note repeats
  the length, citation and offer instruction — the local model follows a per-turn note far better than a standing rule.
  The same block restates the live-facts rule: weather, time/date, news and emails always come from the matching tool,
  never a temperature, time or headline from memory; a failed tool is reported as failed (a real transcript had the
  model invent 27 °C with no tool call).
- **Deeper mode**: `llm` remembers `_deeper_offer = {source, parts}` when a reply ended with the offer (detected on
  the model's output: the offer sentence is produced by a fixed phrase list the prompt asks for). If the next user
  turn is a short affirmative ("yes", "please", "go on", "dig deeper" — small regex, ≤ 6 words), that turn:
  injects the raw chunks for the offered parts ± 1 neighbour (≤ 3 chunks, ≤ 2500 tokens) as a `<source>` block on
  the user message, sets verbosity to `long` for that single completion (the sent system message, not the stored one),
  and clears the offer. Any other message clears the offer too. The parts come from the turn's recall hits, `search_wiki`
  snippets and `read_source` calls; a reply that offers without having used any source leaves no offer.
- **Recall precision**: a chunk is injected only when ≥ 2 distinct query content words match it, or one word that is
  a proper-noun boosted term (the PRD 32 title-case rule, written Title-case ≥ 50 % of the time in the sources) with
  score ≥ `recall_min_score`; single common-word hits ("cook", "today", "egg") no longer pass. Calibration numbers
  from PRD 32 re-run and recorded below.
- Filler: "let me check the guide" style fillers already exist for `search_wiki`; the deeper turn yields a
  `dig_deeper` pseudo tool-call event so the existing filler path speaks one ("Let me read a bit further.").
- No UI change except the Settings reply-length row gaining the note "deeper answers are long regardless".

### Decisions
- Prompt + deterministic "deeper" state over a second model call to classify intent: cheap, testable, local-model safe.
- Offer detection by fixed phrases over structured output: the model already follows short style rules well.
- Assume the model reliably ends with the offer when told; if < 80 % in the eval, add a post-hoc append of the offer
  when a recall/source was used and neighbours exist.

## Test scenarios
1. Recall precision: "what should we cook tonight?", "what's on today?", "any eggs left?" inject nothing against the
   two-guide fixture; "I'm at Covenant and the turrets shoot me" and "Thieves Guild in Riften" still inject.
2. System prompt contains the look-up-first rule and the offer instruction; tool-free runs (eval harness) omit both.
   The block also carries the live-facts rule; a mocked weather turn records the `get_weather` call.
3. Mocked model ends a reply with "Want me to dig deeper?" after a recall turn → `_deeper_offer` set with the source
   and parts; a following "yes please" turn gets the neighbour chunks in its message, verbosity `long` for that
   completion only (assert the max_tokens/style passed), offer cleared; "no, what's the weather" clears it and
   injects nothing; an offer never persists past one turn.
4. Live (marker `live`, the three real guides — Fallout 4, Skyrim, Morrowind): 6 questions; ≥ 5 answers cite the
   guide; ≥ 4 end with the offer; ≥ 4 "yes" turns give an answer ≥ 2× the first one's length that mentions a
   Title-case term from a neighbouring chunk (sampling makes a single follow-up land at 1.5× now and then); "how is the weather today?" calls `get_weather`. Print all
   pairs.
5. Fixture render: Settings reply-length row shows the note.

## Out of scope
Multi-turn research loops, re-ranking with the LLM, changing the default reply length, web sources.

## Results

### Calibration (PRD 32 fixture: two generated FAQs, `tests/test_source_recall.py`)
| | before (PRD 32 rule) | after (precision rule) |
|---|---|---|
| Fallout in-guide questions hitting their section | 10/10 | 10/10 |
| Skyrim in-guide questions hitting their section | 10/10 | 10/10 |
| off-topic questions injecting, default threshold 4.5 | 0/10 | 0/10 |
| off-topic questions injecting, threshold 0 | 9/10 | 4/10 |
| "what should we cook tonight?" injecting, threshold 0 | yes | no |
| "I'm at Covenant and the turrets shoot me" / "Thieves Guild in Riften" | inject | inject |

The default threshold already kept the off-topic set out; the precision rule is what removes the single-common-word
matches that survive a low threshold.

### Live (three real guides, oMLX, `pytest -m live tests/test_grounded_answers.py -s`)
- Prompt wording iterated once: with the offer only in the system block the model cited 5/6 and offered 0/6; with the
  offer and length instruction repeated in the per-turn recall note it cites 6/6 and offers 6/6.
- "Yes" follow-ups: 2.3×–5.6× the short answer, every one naming a term from a neighbouring chunk (e.g. Sellus
  Gravius / Seyda Neen for the Caius Cosades question, Bleak Falls Barrow / Draugr for the Thane question).
- Citations read as a clause inside a sentence ("according to the Fallout 4 FAQ and Walkthrough, …", "per the …").
