Status: refactoring

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
  On the "yes" turn the model names the topic (a first pass whose text is held back, with a per-turn note asking
  for `get_wikipedia(<article name>)`); if it does not call the tool, `get_wikipedia` runs with the question
  itself — a full question as the search string ranks badly ("what exactly is the Kessler syndrome" → a list of
  misconceptions), an article name does not. A 429 from Wikipedia gets one retry after 3 s, then the error string.
- Fillers for `get_wikipedia`.
- Calibration set `tests/fixtures/routing_questions.json`: 8 general (expect direct), 8 filed-document questions
  (expect local; the two generated FAQs mocked, the two real guides live when `~/llm-wiki-dev` has them, else the
  generated FAQs), 8 precise/fresh (expect Wikipedia offer or call), 4 personal ("what did we decide on Falcon") →
  local/none. Run mocked (assert the tool/offer chosen from the model's output shape) and live (marker `live`):
  report a confusion table; iterate on rule wording at least twice; record runs in the PRD.

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

## Results

Live calibration: `pytest -m live tests/test_wikipedia_routing.py -s` — the 28 questions through `llm.ask`
against oMLX (gemma-4-26b), one fresh session each, the two real guides (Fallout 4, Elder Scrolls V) filed into a
temp vault so the filed-document questions get real recall. Observed routes: `direct` (own knowledge, no source, no
offer), `local` (search_wiki/read_source called, or a recall block answered with the dig-deeper offer),
`wikipedia-offer`, `wikipedia-call`, `none` (no source and "I have nothing on that"). `personal` accepts local, none
or direct — only Wikipedia is a miss.

### Run 1 — first wording
Rule 2 "general knowledge you are sure of ... answer directly, no source named and no offer"; rule 3 "asks for a
precise fact — an exact year or date, a number or measurement, the definition of a named term, something from the
last few years — or you are not sure".

| expected \ observed | direct | local | wikipedia-offer | wikipedia-call | none |
|---|---|---|---|---|---|
| direct | 0 | 3 | 5 | 0 | 0 |
| local | 0 | 8 | 0 | 0 | 0 |
| wikipedia | 0 | 0 | 8 | 0 | 0 |
| personal | 0 | 4 | 0 | 0 | 0 |

20/28. Local, Wikipedia and personal perfect; the model offered Wikipedia after confident general answers
(photosynthesis, DNA, crocodile vs alligator, the refrigerator) and, for "capital of France", refused because the
recall block (Fallout's "Capital Wasteland") was off-topic. Two of the three "local" misses were direct answers
with an ignored recall block (the first classifier counted any injected recall as local; fixed for run 2);
"who was Ada Lovelace" was answered from the Fallout guide's robot companion Ada — recall noise from PRD 34's
proper-noun rule. The "yes" follow-up ran `get_wikipedia("photosynthesis")` and answered per Wikipedia at `long`.

### Run 2 — second wording
Rule 2 names the kinds ("who someone was, what a thing is or does, how or why something works, everyday facts like
capitals and authors ... answer directly and stop: no source named, no offer, and a recall block about something
else is ignored, not mentioned"); rule 3 "offer Wikipedia only when the question itself asks for an exact figure (a
specific year or date, a count, a measurement, a precise definition of a named term), or is about the last few
years, or when you truly do not know ... A confident general answer never gets that offer."

| expected \ observed | direct | local | wikipedia-offer | wikipedia-call | none |
|---|---|---|---|---|---|
| direct | 7 | 1 | 0 | 0 | 0 |
| local | 0 | 8 | 0 | 0 | 0 |
| wikipedia | 1 | 0 | 7 | 0 | 0 |
| personal | 0 | 4 | 0 | 0 | 0 |

26/28. Misses: "who was Ada Lovelace" (the Fallout recall again) and "what exactly is the Kessler syndrome"
(a confident direct definition, no offer). The "yes" follow-up ran `get_wikipedia("Ada Lovelace")`.

### Run 3 — final wording (shipped as `llm.ROUTING_RULES`)
Run 2 plus the example phrasing in rule 3: "the precise definition of a named term — \"what exactly is X\"".

| expected \ observed | direct | local | wikipedia-offer | wikipedia-call | none |
|---|---|---|---|---|---|
| direct | 7 | 1 | 0 | 0 | 0 |
| local | 0 | 8 | 0 | 0 | 0 |
| wikipedia | 0 | 0 | 8 | 0 | 0 |
| personal | 0 | 4 | 0 | 0 | 0 |

27/28. The one miss is "who was Ada Lovelace" answered from the Fallout guide (recall noise, a PRD 34 precision
matter, not routing). No filed-document question triggered Wikipedia in any run. The "yes" follow-up ran
`get_wikipedia("Ada Lovelace")` and answered per Wikipedia at `long`.

