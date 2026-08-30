Status: exploration

# 37 — Wikipedia tool and knowledge routing

## Overview
Wanko now has three places an answer can come from: his own knowledge, the local knowledge base (wiki pages and
filed documents, PRD 32/34), and — with this PRD — Wikipedia. The tool is small; the value is in routing: knowing
which source fits a question, saying which one he used, and offering Wikipedia when the local base has nothing and
the question deserves precision. Calibrated with a scripted set through the text path.

## User stories
- As a user, "who was Ada Lovelace" gets a direct answer; "what year exactly did she publish the notes" gets an
  offer or a lookup, with "per Wikipedia" in the answer.
- As a user, a question about something I filed goes to my documents first, never to Wikipedia.
- As a user, when he is unsure or the topic is fresh/precise (dates, numbers, people, places), he offers
  "want me to check Wikipedia?" and a yes fetches it.

## Technical approach
- Tool `get_wikipedia(query)` (tools/wikipedia.py): `action=query&list=search` (top 3) then REST
  `page/summary/<title>` for the best hit → title, one-paragraph extract (≤ 1200 chars), URL; `read_wikipedia
  (title, section)` for a named section via the parse API (≤ 3 k chars). Fixed hosts (`en.wikipedia.org`), 5 s
  timeouts, `User-Agent: voicebot/<version> (personal assistant; +WIKIPEDIA_CONTACT if set)`, 1 req/s throttle,
  in-memory cache per title for the session. Results wrapped as untrusted text once PRD 33 lands (until then plain).
- Routing rules (system prompt, grounding block from PRD 34): 1) if the wiki index / recall names a source on the
  topic → local first; 2) else if the question is general knowledge you are confident about → answer directly, no
  citation; 3) else if it asks for precise facts (dates, numbers, definitions of named things, recent events) or you
  are unsure → answer briefly and end with "Want me to check Wikipedia?" (fixed phrase, reuse the PRD 34 offer
  state so "yes" runs the tool and answers at `long`); 4) never call Wikipedia for anything personal or filed.
- Fillers for `get_wikipedia`.
- Calibration set `tests/fixtures/routing_questions.json`: 8 general (expect direct), 8 filed-document questions
  (expect local; uses the two generated FAQs), 8 precise/fresh (expect Wikipedia offer or call), 4 personal ("what
  did we decide on Falcon") → local/none. Run mocked (assert the tool/offer chosen from the model's output shape) and
  live (marker `live`): report a confusion table; iterate on rule wording at least twice; record runs in the PRD.

### Decisions
- Offer-then-fetch over auto-fetch: a network call per question is slower and the user asked for an offer.
- Same offer state machine as PRD 34: one "yes" path for both "dig deeper" and "check Wikipedia".
- No `fetch_url`: fixed Wikipedia endpoints only (PRD 33 read-only policy).

## Test scenarios
1. Tool: search + summary parsed from canned responses; network error → clear error string, no exception; cache hit
   makes no second request; `read_wikipedia` returns the named section only.
2. Prompt contains the four routing rules only when tools are registered.
3. Offer/yes: a reply ending with "Want me to check Wikipedia?" then "yes" → `get_wikipedia` called with the
   question's topic, answer at `long`, offer cleared; "no" clears without a call.
4. Calibration (mocked): the 28-question set routes as expected ≥ 24/28.
5. Live: same set, printed confusion table, ≥ 20/28 after iteration; a filed-document question never triggers
   Wikipedia.
6. Policy test: the tool's hosts are in the fixed list; no URL argument.

## Out of scope
Offline Wikipedia (Kiwix), other languages, images, auto-fetching without an offer, writing anything.
