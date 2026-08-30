# Stress test 2026-08-30 — documents, Wikipedia, topic switching, UI commands, robustness

Driver: `tests/qa/stress.py` (Playwright, typed turns, replies muted) against `chatbot.py` on :8014 with a
throwaway vault: the Fallout 4, Skyrim and Morrowind guides, three EPUBs (The Art of Decoding Microservices,
Crafting Great APIs with DDD, The Art of Immutable Architecture), plus `people/anna` and `projects/falcon`.
Model gemma-4-26B via oMLX. Facts were checked by grepping the raw files; "recall" below is the offline
`llm.recall_for` prediction for the same question against the same vault (what the turn injected).
Full transcript: `/tmp/qa/transcript.md` (not committed).

## Scorecard

| section | turns | correct | partial | wrong | notes |
|---|---|---|---|---|---|
| A documents | 14 | 11 | 0 | 3 | both "not covered" questions handled; both "yes" follow-ups stayed on the right guide |
| B Wikipedia | 6 | 2 | 1 | 3 | offer → yes → `get_wikipedia` works; local-first routing misses |
| C topic switching | 7 | 6 | 0 | 1 | pronoun follow-up and `/new` summary fine |
| D UI instructions | 8 | 6 | 2 | 0 | find / back / open / palette all deterministic |
| E robustness | 5 | 4 | 1 | 0 | no server errors, no prompt leak, order kept |
| **total** | **40** | **29** | **4** | **7** | |

Timing (39 timed turns): median 6.8 s, mean 7.4 s, p90 11.6 s, max 19.3 s (the two "dig deeper" turns:
17.0 s and 19.3 s; `get_wikipedia` turn 14.1 s). Whole battery 288 s of turn time.
Context: max **37 %** (after 24 document turns, before `/new`); `/new` dropped it to 21 %; never trimmed.
Zero `[Error]`/traceback lines in the server log; zero `search_wiki` / `read_source` calls in 38 turns.

## Worst transcripts (trimmed)

1. **A8** "Who trains Unarmored in Morrowind and up to what level?" → *"the provided documents do not mention who
   trains Unarmored…"* — **wrong/refused**. Guide line 1908: "Top Trainers: Caius Cosades, Balmora (up to 70);
   Khargol gro-Boguk … (up to 100)". Recall injected parts 43 (Mudcrab leveling) and 174 (Seyda Neen); the
   trainer text is in parts 34/41/42.
2. **A10** "…what trade-off does the author describe for a database per service?" → *"the snippet … only
   contains the cover and introduction"* — **wrong/refused**. Book has a "Database Per Service" entry
   ("Managing data consistency and integrity across services, often requiring data replication or event
   sourcing"), parts 92/93; recall injected part 1 (the cover).
3. **A11** "…what signals a bounded context at the end of an Event Storming workshop?" → *"bounded contexts and
   team boundaries are defined by the context map"* — **hallucinated**. Book: "indicated by a change in role …
   or by a change in aggregate". Recall injected part 131 (a list of GitHub URLs).
4. **B6** "What did we decide on Falcon?" → *"I don't have any recorded decisions regarding the Falcon project…
   I'll check the wiki…"* (no tool call) — **wrong**. `projects/falcon` Decisions: "ship Falcon on Postgres, not on
   Kafka event sourcing". Pages are never auto-recalled and the model only promised to search.
5. **B5** "Who is Vivec?" → *"I don't have any information about Vivec in my files. Would you like me to check
   Wikipedia?"* — **wrong**. Morrowind guide has "Vivec & Wilderness {WLK007}"; three words < `RECALL_MIN_WORDS`.
6. **B1** "Who was Ada Lovelace?" → *"In Fallout 4, Ada is … a survivor robot … Want me to check Wikipedia?"* —
   **wrong** (the PRD 37 known miss, still there): recall parts 488/492 on the single proper term "Ada".
7. **C3** "What does The Art of Decoding Microservices say about the saga pattern?" → *"the provided sections …
   do not mention the saga pattern"* — **wrong**. Saga appears in 8 chunks (e.g. "Saga Pattern: Implementing a
   sequence of local transactions… compensating transactions"); recall injected parts 1 (cover) and 172 (recap).
8. **B4** "What happened in the 2024 Paris Olympics opening ceremony?" → called `get_news`, *"my current news
   feed doesn't have any information…"* — **partial**: no wrong fact, but the routing rule says a last-few-years
   question gets the Wikipedia offer, not the headline feed.
9. **D3** Sources view, Morrowind guide open, "summarise this" → *"This part of the Morrowind guide covers …
   Craftsmen's Hall and the Residential and Temple Sewers … Mournhold…"* — **partial**: it summarised one random
   chunk (BM25 on "summarise this") instead of the page head/Summary.
10. **E2** 1336-char event-sourcing question ("is event sourcing a good fit for a billing system?") → two
    sentences from own knowledge + *"Want me to check Wikipedia?"* — **partial**: three filed books cover it
    (no recall injected) and a design opinion gets the Wikipedia offer.

Also: **D4** "what does it say about Caius Cosades?" added "at the South Wall Cornerclub in Balmora", which
the recalled chunk does not say (the guide never places Caius there) — partial.

## Defects, ranked

1. **Recall ranks the source's own title words.** A question that names the guide/book ("in the Morrowind
   guide", "The Art of Decoding Microservices") matches every chunk that repeats the title (cover, TOC, running
   headers) and, with proper-noun boosting, those outrank the answer chunk; `RECALL_MAX_CHUNKS = 2` leaves no
   room for the real hit (A8, A10, A11, C3 — 4 of 7 wrong answers). Fix: in `llm.recall_for`
   (`llm.py:587`) / `wiki.search_chunks` (`wiki.py:1072`) drop query terms that are tokens of a filed source's
   title before scoring (keep them only for choosing which source), and dedupe hits per source with `k=4`.
2. **A recall miss is final: the model never calls `search_wiki`/`read_source`.** 0 calls in 38 turns; every
   miss became "the provided documents do not mention…". The look-up-first rule (`GROUNDING_RULES`,
   `llm.py:173`) is not followed. Fix: make it deterministic like PRD 38 — when recall injects nothing for a
   question with a proper term, or the reply contains "do not mention / don't have anything", run
   `search_wiki(question)` server-side and re-ask once with the snippets (`llm.ask_events`, `llm.py:609-627`).
3. **Wiki pages are never auto-recalled.** `recall_for` only searches raw chunks; `projects/falcon` with a
   Decisions section is invisible unless the model calls a tool (B6, and it claimed to check without calling).
   Fix: `llm.py:580` — add the top `wiki.search(question, 1)` page hit (Summary + best section, ≤ 400 tokens)
   to the recall block when a proper term matches the page name/title.
4. **Short questions never recall** (`RECALL_MIN_WORDS = 4`, `llm.py:398`): "Who is Vivec?" gets "nothing in my
   files" plus a Wikipedia offer although Vivec is a guide chapter. Fix: allow ≥ 2 words when
   `wiki.proper_terms(terms)` is non-empty.
5. **Single-proper-term noise: "Ada Lovelace" → Fallout's robot Ada** (B1, unchanged since PRD 37). Fix: in
   `recall_for` (`llm.py:592`) require the chunk to match the full proper bigram when the question has one
   ("Ada Lovelace"), else treat it as a common-word match needing ≥ 2 terms.
6. **Last-few-years questions go to `get_news`** (B4). Fix: `ROUTING_RULES` (`llm.py:191`): "the news tool is
   for today's headlines only; a past event of the last few years gets the Wikipedia offer".
7. **"summarise this" on a source quotes a random chunk** (D3). `focus_block` (`llm.py:472`) always calls
   `wiki.best_chunk(path, question)`; for a content-free question (`wiki.content_terms(question)` empty)
   skip the chunk and send the head + Summary only.
8. **Filed books ignored for design questions + Wikipedia offer on an opinion** (E2). The long repeated text
   diluted the per-word score; fix: dedupe query terms in `wiki.search_chunks` before normalising, and add
   to `ROUTING_RULES` that advice/opinion questions never get the offer.
9. **Double date in raw file names**: dropping `2026-08-30-note.md` files it as
   `raw/inbox/2026-08-30-2026-08-30-….md`. Fix: `sources.ingest_upload` (`sources.py:1427`) strip a leading
   `\d{4}-\d{2}-\d{2}-` from `stem` before `_dated()`.
10. **"open the window please" → "no page for 'the window please'"**: `_OPEN_PAGE` (`chatbot.py:343`) strips a
    leading "please" only. Fix: also strip a trailing `,? please`. (The no-match remark itself is acceptable.)
11. Deeper answers embellish (D4 "South Wall Cornerclub"): the deeper/focus note could add "only what the
    passage says; do not add locations or names from memory" (`FOCUS_NOTE`, `llm.py`).

## What held up

- Recall precision on the guides when the question does not name the source: 9/9 correct with the source
  named in a clause and the "Want me to dig deeper?" offer; both "yes" turns read the right neighbours
  (Honest Dan / Human Error; Farengar / skill book Thief) and never crossed guides.
- Both uncovered questions (A13, A14) were declined honestly; Wikipedia offer → yes → `get_wikipedia` → "per
  Wikipedia" (B2/B3); pronoun follow-up after two topic hops went back to the Fallout DLC (C4, Automatron $10
  / Far Harbor $25 both per the guide); `/new` and the palette "New topic" produced accurate summaries and
  "what were we talking about?" reflected them; the 90 % notice was never needed (max 37 %).
- Find / go back / open / ⌘K page open switched views deterministically with a one-line remark; the
  regex no-match path answered without inventing a page; empty send is ignored; rapid double-send and a
  send during "thinking…" were both queued and answered in order; the `</document>` injection was refused
  without leaking the prompt.
