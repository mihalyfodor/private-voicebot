Status: implementation

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
- **Prompt rules** (system prompt, character-independent block): if the wiki index or a `<recall>` block names a
  source on the topic, call `search_wiki`/`read_source` (or use the recall) before answering from memory; cite the
  source title in one clause; when the answer comes from a document and more is available (`part n/N` neighbours,
  more hits), end with a short offer ("Want me to dig deeper?"). Off-topic recall is to be ignored silently.
- **Deeper mode**: `llm` remembers `_deeper_offer = {source, parts}` when a reply ended with the offer (detected on
  the model's output: the offer sentence is produced by a fixed phrase list the prompt asks for). If the next user
  turn is a short affirmative ("yes", "please", "go on", "dig deeper" — small regex, ≤ 6 words), that turn:
  injects `read_source` text for the offered parts ± 1 neighbour (≤ 3 chunks, ≤ 2500 tokens), sets verbosity to
  `long` for that single completion, and clears the offer. Any other message clears the offer too.
- **Recall precision**: a chunk is injected only when ≥ 2 distinct query content words match it, or one word that is
  a proper-noun boosted term (the PRD 32 title-case rule) with score ≥ `recall_min_score`; single common-word hits
  ("cook", "today", "egg") no longer pass. Calibration numbers from PRD 32 re-run and recorded.
- Filler: "let me check the guide" style fillers already exist for `search_wiki`; add one for the deeper turn.
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
3. Mocked model ends a reply with "Want me to dig deeper?" after a recall turn → `_deeper_offer` set with the source
   and parts; a following "yes please" turn gets the neighbour chunks in its message, verbosity `long` for that
   completion only (assert the max_tokens/style passed), offer cleared; "no, what's the weather" clears it and
   injects nothing; an offer never persists past one turn.
4. Live (marker `live`, both guides): 6 questions; ≥ 5 answers cite the guide; ≥ 4 end with the offer; "yes" gives an
   answer ≥ 2× the first one's length that mentions a term from a neighbouring chunk. Print all pairs.
5. Fixture render: Settings reply-length row shows the note.

## Out of scope
Multi-turn research loops, re-ranking with the LLM, changing the default reply length, web sources.
