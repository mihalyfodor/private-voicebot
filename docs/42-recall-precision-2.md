Status: done

# 42 — Recall precision 2: the stress-test defects

## Overview
The 2026-08-30 stress test (`docs/reviews/2026-08-30-stress-test.md`) scored 29/40 with seven wrong answers, and
every one of them traces to recall: the source's own title words outrank the answer chunk, a miss is final because
the model never calls `search_wiki`, wiki pages are never auto-recalled, three-word questions never recall, and a
lone proper noun ("Ada") matches the wrong document. This PRD fixes the report's defects 1–5 deterministically on
the server side and takes the cheap extras it lists (6–11), then re-runs the same battery.

## User stories
- As a user, "in the Morrowind guide, who trains Unarmored?" or "what does The Art of Decoding Microservices say
  about the saga pattern?" answers from the passage that holds the answer, not from the cover or the contents page.
- As a user, when Wanko is about to say "the documents do not mention it" although a filed document does, it
  looks it up once more on its own and answers from what it finds; it never claims to have checked without checking.
- As a user, "what did we decide on Falcon?" answers from the `projects/falcon` page's Decisions without a tool call.
- As a user, "who is Vivec?" answers from the Morrowind guide, not with a Wikipedia offer.
- As a user, "who was Ada Lovelace?" is a direct answer; the Fallout robot Ada stays out of it.
- As a user, "summarise this" with a source open summarises the source, not a random chunk of it.
- As a user, a question about an event of the last few years gets the Wikipedia offer, not the news feed; a design
  opinion ("is event sourcing a good fit for billing?") gets an answer, not a Wikipedia offer.
- As a user, "open the window please" does not look for a page called "the window please".
- As a developer, dropping `2026-08-30-note.md` files it as `raw/inbox/2026-08-30-note.md`, not with the date twice.

## UI/UX
No new controls. One visible change: when the fallback fires, the first sentence of the reply is held back for a
moment (the status stays "thinking…" a little longer) and the reply that arrives is the grounded one; nothing is
spoken twice. Everything else is a different (better) answer to the same turn.

## Technical approach

### 1. Title words never rank a chunk; four candidates, no overlapping pair (defect 1)
- `wiki.search_chunks(query, k)`: the query's content terms are split into *title terms* (tokens of any filed
  source's title, e.g. `art decod microservice`, `morrowind game year edition guide`) and *topic terms* (the rest).
  When topic terms remain, only they are scored and normalised (`score / len(topic terms)`, `matched` lists topic
  terms). When the question names a source — ≥ 2 of one title's tokens in the question, or one token that no other
  title has and is not a stopword-like generic (`guide`, `note`, `book`…) — the hits are restricted to the named
  sources, and a named source is quoted without the `recall_min_score` threshold — naming the document settles
  relevance, which is all the threshold decides (the precision rule still applies). Question filler ("does", "up",
  "off", "per", "according") joins the stopwords and the generic words ("say", "describe"…) are not normalised on,
  so "what does the book say about X" is scored on X. A question made only of title words ("what is the Morrowind
  guide about") is scored as before.
- `llm.recall_for` asks for `k = 4` candidates and keeps at most `RECALL_MAX_CHUNKS = 2`, skipping a candidate that
  is the neighbour (`n ± 1`, same source) of one already kept — neighbours share their overlap text and the deeper
  turn reads them anyway.

### 2. The deterministic fallback (defect 2)
- `llm.ask_events` holds the first `FALLBACK_HOLD_CHARS = 240` characters of a first-pass text reply (no tool
  call, not a deeper / Wikipedia-yes turn, no page focused) before yielding them. If the held text matches
  `NOT_COVERED` ("do not / does not / don't mention", "not covered", "no information", "nothing in my files",
  "documents don't cover"…) the server runs the `search_wiki` equivalent itself: `wiki.search(question, 1)` plus
  `wiki.search_chunks(question, k=3)` minus the parts already injected, filtered by the PRD 34 precision rule.
  With at least one hit, the held reply is discarded (never appended, never spoken), a `<lookup>` block
  (≤ `FALLBACK_MAX_TOKENS = 1500`, wrapped as untrusted text) with the note `FALLBACK_NOTE` ("a search of the
  filed documents found these passages; answer from them if they cover the question, name the source once, end
  with the deeper offer; otherwise ignore them completely and answer as you normally would, never mentioning
  documents or what you could not find" — the first battery run showed "the documents do not cover a modular
  monolith" on a general-knowledge follow-up when the note asked for that sentence) is appended to the user message and the
  turn is asked once more; the second reply is streamed as the turn's reply. Without hits the held text is
  flushed and streaming continues. One retry per turn, never after a tool round.
- A question with no recall at all and a proper term is covered by the same path: the model's "I don't have
  anything" triggers the lookup, which is the recall search without the threshold.

### 3. Wiki pages join auto-recall (defect 3)
- `recall_for` also runs `wiki.search(question, 1)`. The top page joins the block when every token of the page's
  name (`name:` frontmatter, else the path stem) is among the question's content terms and no `COMMAND_WORDS`
  term is (an email/news/weather/time command about Anna stays a tool command). The page contributes its head +
  `## Summary` and the `##` section that best matches the question (`wiki.best_section`; Decisions for "what did we
  decide"), as `From page "Falcon" (projects/falcon): …`, ≤ `RECALL_PAGE_TOKENS = 400`, ahead of the chunks
  within `RECALL_MAX_TOKENS`.

### 4. Short questions recall on a proper term (defect 4)
- `recall_for`: a question under `RECALL_MIN_WORDS` still searches when it has ≥ `RECALL_MIN_PROPER_WORDS = 2`
  words and `wiki.proper_terms(terms)` is non-empty ("Who is Vivec?"). Small talk ("any eggs left?") is unchanged.

### 5. A proper bigram must match whole (defect 5)
- `wiki.proper_phrases(query)`: consecutive query tokens where one is a proper term and the other is a proper
  term, a word the corpus has never seen ("Lovelace"), or both are Title-case in the text. `search_chunks` reports
  per hit which phrases occur adjacent in the chunk (`phrases`). In `recall_for` (and the fallback) a proper term
  that belongs to a phrase the chunk does not contain whole is demoted to a common word: the hit then needs ≥ 2
  matched terms. "Ada Lovelace" never matches lone "Ada"; "Caius Cosades" still matches "Caius Cosades".

### 6. "summarise this" on a source (defect 7)
- `wiki.topic_terms(question)` = content terms minus `GENERIC_WORDS` (summarise, summarize, summary, explain,
  describe, overview, gist, page, document, source, say, says, tell…). `focus_block` skips the best chunk / section
  when it is empty: the block is the page head + `## Summary` only.

### 7. Routing wording (defects 6, 8, 11)
- `ROUTING_RULES`: "the news tool is for today's headlines only: a past event of the last few years gets the
  Wikipedia offer unless the question asks for the news"; "advice or opinion questions — should I, is X a good fit,
  what would you do — never get the offer". The tool line in the base prompt reads "asked for the news or today's
  headlines".
- `FOCUS_NOTE` and `DEEPER_NOTE` add "only what the passage says; do not add locations or names from memory".

### 8. Open-page politeness (defect 10)
- `chatbot._OPEN_PAGE` drops a trailing `,? please` from the query as it already drops a leading one.

### 9. Raw file names (defect 9)
- `sources._dated(slug)` strips one leading `YYYY-MM-DD-` from the slug before stamping today's date.

### Decisions
- Chose a server-side held-reply fallback over teaching the model to call `search_wiki`: 0 tool calls in 38 turns
  says the prompt rule does not work with this model; PRD 38 already made focus deterministic for the same reason.
- Hold the first 240 characters rather than the whole reply: the claim of non-coverage is in the first sentence,
  and sentence-wise TTS starts no earlier anyway.
- Title terms are dropped from scoring but kept for choosing the source: a named source is a filter, not evidence.
- Pages join recall by name match, not by BM25 alone: a page hit on a common word ("deploy") would be noise.
- Assume ≤ 1 retry per turn keeps the p90 under 15 s; if the fallback fires on most turns, the threshold is wrong,
  not the retry.
- Risk: the phrase list misses a wording of "not covered" — mitigation: the fixture-driven unit test lists the
  transcripts' actual phrasings; the battery records how often the fallback fired.

## Data model / state
- No persisted change. `_turn_sources` also records fallback hits (the deeper offer works after a fallback).
- `search_chunks` hits gain `phrases` (list of matched proper bigrams). The chunk cache keeps the corpus
  vocabulary (`_chunk_cache["proper"]` has an entry — 0.0 or more — for every stem the sources contain).
- `raw/manifest.jsonl` paths for dated uploads carry one date.

## Test scenarios
1. Title stripping: against the two generated FAQs, `search_chunks("in the Fallout 4 guide, what about Covenant
   turrets")` scores `covenant turret` only, restricts to the Fallout guide, and its top hit is a Covenant chunk;
   `recall_for` with the question naming the guide quotes a Covenant chunk, never the contents page (part 1).
   `search_chunks("what is the Fallout 4 guide about")` still returns hits. Observable: hit `matched` lists no
   title tokens; part 1 absent. With the threshold at 999, a question naming the guide still quotes it
   (`named` hits ignore the floor) while the same question without the title injects nothing.
2. Neighbour dedupe: with the threshold at 0 and k = 4 candidates, `recall_for` never quotes two chunks `n` and
   `n + 1` of the same source and still quotes up to 2. Observable: the part numbers in the block differ by ≥ 2.
3. Fallback: a mocked first reply "[sad] The provided documents do not mention Covenant turrets." to a Covenant
   question with recall disabled (threshold 999) → the events carry no delta of that text, a `<lookup>` block with a
   Covenant chunk rides on the user message of the second call, the second reply is the streamed one, the stored
   assistant turn is the second reply, `_deeper_offer` points at the guide when it ends with the offer. A first
   reply "[happy] Talk to Jacob." streams normally (one call). A "not covered" reply about a topic no source has
   ("submarine periscope") streams as is (one call). A tool-call turn never retries.
4. Page recall: seeded vault with a Falcon page that has a Decisions section; `recall_for("What did we decide on
   Falcon?")` contains `From page "Falcon" (projects/falcon)`, the Summary and the Decisions line, ≤ 400 tokens;
   "did I get any email from Anna today" injects nothing; a page never joins on a common word alone.
5. Short proper question: `recall_for("Who is Covenant?")` (3 words, proper) quotes the Fallout guide;
   `recall_for("any eggs left?")` and `recall_for("hi")` stay empty.
6. Proper bigram: `wiki.proper_phrases("who was Covenant Lovelace")` = [("covenant", "lovelace")];
   `recall_for("who was Covenant Lovelace")` is empty although lone "Covenant" chunks exist; "Honest Dan at
   Covenant" still recalls (the bigram "Honest Dan" occurs in the chunk).
7. Summarise focus: with a source focused, `focus_block("summarise this")` has the source page head + Summary and
   no `Part n/N` chunk; `focus_block("what about Covenant turrets")` still has the chunk.
8. Routing wording: the system prompt contains "today's headlines only" and "never get the offer" for opinions;
   `FOCUS_NOTE` and `DEEPER_NOTE` contain "do not add locations or names from memory". The mocked routing
   calibration stays ≥ 24/28 with every `local` question local.
9. `chatbot.page_command("open the window please")` == ("open", "window") (the leading article was always
   dropped, PRD 38); "open Anna, please" == ("open", "Anna"); "please open Anna" unchanged.
10. `sources.ingest_upload(b"...", "2026-08-30-note.md")` files `raw/inbox/<today>-note.md`; a name without a
    date is unchanged.
11. Regression: PRD 32/34/37/38 unit suites pass unchanged except the OFF_TOPIC calibration, which now asserts no
    *chunk* is quoted (a page about Anna joining "remind me what Anna said in the meeting" is the intended new
    behaviour).
12. Battery: `tests/qa/stress.py` (same 40 turns, same vault recipe, gemma-4-26B via oMLX) graded with the same
    rubric, recorded below with the per-section table. Target ≥ 35/40 correct, no new wrong answers in sections
    C–E, zero server errors.

## Out of scope
- Embeddings or a re-ranker; changing `RECALL_MIN_SCORE`; more than one retry per turn.
- Teaching the model to call `search_wiki` itself (the prompt rule stays, the server no longer depends on it).
- A second battery for the EPUB-only questions; new UI.

## Results

Battery: `tests/qa/stress.py`, same 40 turns and vault recipe as the 2026-08-30 review (three guides, three EPUBs,
`people/anna`, `projects/falcon`), gemma-4-26B via oMLX, chatbot.py on :8014 with a throwaway vault under /tmp/qa.
Graded by hand against the raw files with the review's rubric. Two runs after the change: run 1 with the first
`FALLBACK_NOTE` wording ("if they do not cover it, say the documents do not cover it"), run 2 after the one
iteration (the note now says to ignore the passages completely and answer as usual, never mentioning documents).

| section | turns | before (review) c / p / w | run 1 c / p / w | run 2 c / p / w |
|---|---|---|---|---|
| A documents | 14 | 11 / 0 / 3 | 12 / 1 / 1 | 12 / 1 / 1 |
| B Wikipedia | 6 | 2 / 1 / 3 | 4 / 1 / 1 | 5 / 1 / 0 |
| C topic switching | 7 | 6 / 0 / 1 | 7 / 0 / 0 | 7 / 0 / 0 |
| D UI instructions | 8 | 6 / 2 / 0 | 7 / 1 / 0 | 6 / 2 / 0 |
| E robustness | 5 | 4 / 1 / 0 | 3 / 2 / 0 | 4 / 1 / 0 |
| **total** | **40** | **29 / 4 / 7** | **33 / 5 / 2** | **34 / 5 / 1** |

Run 2 is one correct short of the 35 target; the seven wrong answers of the review are down to one (A11). Fixed
outright: A8 (Unarmored trainers, part 34), A10 (database per service, via the fallback), C3 (saga pattern),
B6 (Falcon decision from the page), B5 (Vivec from the guide), B1 (Ada Lovelace direct, no robot Ada, no offer),
E2 (opinion answered, no Wikipedia offer). Timing run 2 (39 timed turns): median 6.2 s, mean 6.6 s, p90 11.3 s,
max 13.9 s (run 1: 6.3 / 6.5 / 9.7 / 16.7; review: 6.8 / 7.4 / 11.6 / 19.3). Max context 36 %, never trimmed,
zero `[Error]`/traceback lines both runs. The fallback fired 4 times in run 1 (A10, A13, A14, B4) and 3 in run 2
(A10, A14, D8). The seeded manifest carried a single date per raw name (defect 9).

Still not correct after run 2 (trimmed):
- **A11** wrong: *"Per Crafting Great APIs with Domain-Driven Design, the bounded contexts are located around
  the domain events found during the workshop."* — the book says "a change in role … or a change in aggregate";
  recall quoted parts 182 and 20 (workshop logistics), not the passage.
- **A14** partial: *"I don't have any information regarding the Dawnguard vampire lord perk tree. Want me to
  check Wikipedia?"* — right refusal, but a filed-document question got the Wikipedia offer.
- **B4** partial: *"I don't have specific details on the events of the 2024 Paris Olympics opening ceremony in my
  current files. Since it occurred a while ago, would you like me to check Wikipedia for a summary?"* — no
  `get_news` any more (defect 6 fixed), but a free-form offer instead of the fixed phrase, so a "yes" would not
  run the tool.
- **D2** partial: page reopened correctly (`projects/falcon`), remark *"I have the Decisions page pulled up for
  you. It appears to be empty at the moment."* is wrong.
- **D4** partial: *"…directions to Balmora, where you are instructed to ask at the South Wall Cornerclub."* — the
  guide never places Caius there (unchanged from the review despite the "no names from memory" note).
- **E3** partial: second reply *"The provided document does not contain information about a modular monolith. It
  focuses on the concept of scalability within microservices instead."* — "modular monolith" now recalls a
  monolith chunk (two matched words) and the model reported the mismatch instead of ignoring the block. New
  since the review. After run 2 `RECALL_NOTE` was given the same "ignore it completely, never mention the
  document" wording as `FALLBACK_NOTE`; not re-run live.
