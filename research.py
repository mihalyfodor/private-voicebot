"""Research mode (docs/54-vault-map-research.md): one explicit question run as a fixed, budgeted chain
over the vault and its sources — plan → search → judge → read → write → verify — ending in a card whose
quotes were each checked against the text this run read.

Every step is a PRD 51 card step (live on the card through `notify`) and a PRD 49 `step` span (the run
page's timeline); the model calls go through `core.llm` with the purposes `research_plan`,
`research_judge` and `research_write`. The governor caps sub-questions (4), reads (6), model calls (8)
and wall clock (90 s): on exhaustion the run skips to Write with what it read and ends `partial`, or
`nothing` when nothing verifiable is left. Spans carry counts and titles only, never the question.

A root module until PRD 50 step 9 creates the `agent/` layer; it moves to `agent/research.py` then.
"""
import re
import time

import llm
from agent import recall
from memory import ops
import runs
from vault import chunks as v_chunks
from vault import focus
from vault import search as v_search
from vault import store
from core import llm as core_llm
from core import trace
from core.untrusted import DOCUMENT_RULE, untrusted

MAX_SUBQUESTIONS = 4
RESEARCH_MAX_READS = 6
MAX_CALLS = 8                     # plan 1 + judge 1 + write 1 + up to 5 retries (the judge's fallback)
MAX_SECONDS = 90.0
PAGE_HITS, CHUNK_HITS = 5, 3      # per sub-question
SNIPPET_CHARS = 300
PLAN_TOKENS, JUDGE_TOKENS = 200, 700
LABEL_CHARS = 80
LINE = "Researching that — written out below."
NOTHING_FOUND = "Nothing in the wiki or the sources answers this."

PLAN_PROMPT = (
    "Split the user's research question into at most 4 short search queries for a personal wiki and its "
    "filed documents. Reply with a JSON list and nothing else: each item a string, or an object "
    '{"q": "...", "type": "project"} when the query is about one page type (person, project, meeting, note).')
JUDGE_PROMPT = (
    "You judge search hits for a research question. For every numbered hit decide whether it helps answer "
    "its sub-question. Reply with a JSON list and nothing else, one object per hit: "
    '{"id": <number>, "verdict": "keep" or "drop", "reason": "<one short line>"}. ' + DOCUMENT_RULE)
WRITE_PROMPT = (
    "You write research notes in Markdown from the material given, and only from it. Start with one "
    "`# Title` line naming the topic, then a `## Findings` section of bullets. Every bullet states one "
    'finding and ends with one short quote copied word for word from the material, then its source: '
    '`"exact words" — <source name as given>`. No other sections, no sources list (it is appended for you), '
    "no names or figures from memory. " + DOCUMENT_RULE)


def clock() -> float:
    """The governor's wall clock (tests patch this, not `time.monotonic`)."""
    return time.monotonic()


class Budget:
    """The governor: counts model calls and reads against the caps and watches the wall clock;
    `reason` is the first cap that cut the run ("budget: time" / "budget: reads"). The call cap cannot cut a
    run: plan 1 + judge ≤ 5 (a batch, then one per sub-question) + write 1 stays under MAX_CALLS, and
    `call_left` keeps the write's call in reserve."""

    def __init__(self):
        self.t0, self.calls, self.reads, self.reason = clock(), 0, 0, ""

    def time_left(self) -> bool:
        """Is the wall clock under MAX_SECONDS (else the run is cut with "budget: time")?"""
        if clock() - self.t0 <= MAX_SECONDS:
            return True
        self.cut("budget: time")
        return False

    def call_left(self, reserve: int = 0) -> bool:
        """Is a model call left once `reserve` calls are kept back?"""
        return self.calls < MAX_CALLS - reserve

    def cut(self, reason: str) -> None:
        """Record why the run was cut (the first reason wins)."""
        self.reason = self.reason or reason


def _flat(text) -> str:
    """Whitespace collapsed to single spaces (None is "")."""
    return " ".join(str(text or "").split())


def _complete(budget: Budget, purpose: str, system: str, user: str, max_tokens: int) -> str:
    budget.calls += 1
    return llm.strip_tag(core_llm.complete(system, user, max_tokens, purpose=purpose)).strip()


# ------------------------------------------------------------------ plan


def parse_plan(raw: str, question: str) -> list:
    """``[{q, filters}]`` from the plan reply: at most MAX_SUBQUESTIONS (extra dropped), duplicates and
    empties out; an unparseable or empty reply is the question itself."""
    data = ops.extract_json(raw)
    out = []
    for item in data if isinstance(data, list) else []:
        if isinstance(item, dict):
            q = _flat(item.get("q") or item.get("question"))
            kind = str(item.get("type") or "").strip().lower()
            filters = {"type": kind} if kind in ("person", "project", "meeting", "note", "source") else None
        else:
            q, filters = _flat(item), None
        if q and q.lower() not in {s["q"].lower() for s in out}:
            out.append({"q": q, "filters": filters})
    return out[:MAX_SUBQUESTIONS] or [{"q": _flat(question), "filters": None}]


# ------------------------------------------------------------------ search


def _window(text: str, query: str) -> str:
    """SNIPPET_CHARS of `text` around the first content word of `query` (whitespace collapsed)."""
    flat = _flat(text)
    lower = flat.lower()
    found = [lower.find(t) for t in v_chunks.content_terms(query) if lower.find(t) >= 0]
    start = max(0, min(found) - 60) if found else 0
    return flat[start:start + SNIPPET_CHARS]


def search(subs: list) -> list:
    """Per sub-question the top pages and source chunks (no model call): ``[{q, ref, kind, …}]``, a ref
    only once (under the first sub-question that found it)."""
    hits, seen = [], set()
    for i, sub in enumerate(subs):
        found = 0
        with trace.span("step", name="search") as span:
            for h in v_search.search(sub["q"], PAGE_HITS, sub["filters"]):
                if h["path"] in seen:
                    continue
                seen.add(h["path"])
                hits.append({"q": i, "ref": h["path"], "kind": "page", "path": h["path"],
                             "title": focus.page_title(h["path"]) or h["path"], "score": h["score"],
                             "snippet": _flat(h["snippet"])[:SNIPPET_CHARS]})
                found += 1
            for h in v_chunks.search_chunks(sub["q"], CHUNK_HITS):
                ref = f"{h['title']} · part {h['n']}/{h['total']}"
                if ref in seen:
                    continue
                seen.add(ref)
                hits.append({"q": i, "ref": ref, "kind": "chunk", "source": h["source"], "n": h["n"], "total": h["total"],
                             "title": h["title"], "score": h["score"], "snippet": _window(h["text"], sub["q"])})
                found += 1
            span.update(count=found)
    return hits


# ------------------------------------------------------------------ judge


def _judge_prompt(question: str, subs: list, hits: list) -> str:
    lines = [f"Research question: {question}", ""]
    for i, sub in enumerate(subs):
        lines.append(f"Sub-question {i + 1}: {sub['q']}")
        for n, h in enumerate(hits):
            if h["q"] == i:
                lines.append(untrusted(f"[{n + 1}] {h['ref']} — {h['snippet']}", h["title"]))
        lines.append("")
    return "\n".join(lines)


def _verdicts(raw: str) -> dict:
    """hit number -> (keep, reason) from a judge reply; {} when it does not parse."""
    data = ops.extract_json(raw)
    out = {}
    for item in data if isinstance(data, list) else []:
        if not isinstance(item, dict):
            continue
        try:
            n = int(item.get("id"))
        except (TypeError, ValueError):
            continue
        verdict = item.get("verdict", item.get("keep"))
        keep = verdict is True or str(verdict).strip().lower() in ("keep", "yes", "true")
        out[n] = (keep, _flat(item.get("reason"))[:200] or ("kept" if keep else "dropped"))
    return out


def judge(budget: Budget, question: str, subs: list, hits: list) -> None:
    """One batched completion over every hit (keep|drop + a reason each); a reply that does not parse
    falls back to one call per sub-question. A hit left unjudged is dropped ("not judged")."""
    verdicts = {}
    if hits and budget.time_left() and budget.call_left(reserve=1):
        verdicts = _verdicts(_complete(budget, "research_judge", JUDGE_PROMPT,
                                       _judge_prompt(question, subs, hits), JUDGE_TOKENS))
        if not verdicts:
            for i in sorted({h["q"] for h in hits}):
                if not (budget.time_left() and budget.call_left(reserve=1)):
                    break
                mine = [h for h in hits if h["q"] == i]
                got = _verdicts(_complete(budget, "research_judge", JUDGE_PROMPT,
                                          _judge_prompt(question, subs, mine), JUDGE_TOKENS))
                verdicts.update({hits.index(mine[n - 1]) + 1: v for n, v in got.items() if 1 <= n <= len(mine)})
    for n, h in enumerate(hits):
        h["keep"], h["reason"] = verdicts.get(n + 1, (False, "not judged"))


# ------------------------------------------------------------------ read


def _parts(source: str, n: int, text: str) -> dict:
    """part number -> its chunk text, for part n and the neighbours `read_source` returned with it (the
    verifier re-cites a quote found in a neighbour)."""
    try:
        chunks = v_chunks.raw_chunks(source)["chunks"]
    except KeyError:
        return {n: text}
    return {k: chunks[k - 1]["text"] for k in (n - 1, n, n + 1) if 1 <= k <= len(chunks)}


def read(budget: Budget, hits: list) -> list:
    """The kept hits, best score first, read whole (a chunk with its neighbours, a page ≤ 4 kB) up to
    RESEARCH_MAX_READS: ``[{label, text, kind, source|path, title, n}]``."""
    kept = sorted((h for h in hits if h["keep"]), key=lambda h: (-h["score"], h["ref"]))
    material = []
    for h in kept:
        if budget.reads >= RESEARCH_MAX_READS:
            budget.cut("budget: reads")
            break
        if not budget.time_left():
            break
        budget.reads += 1
        if h["kind"] == "chunk":
            text = v_chunks.read_source(h["source"], h["n"])
            text = text.split("\n", 1)[1] if "\n" in text else text
            material.append({"label": h["ref"], "text": text, "kind": "chunk", "source": h["source"],
                             "title": h["title"], "n": h["n"], "total": h["total"],
                             "parts": _parts(h["source"], h["n"], text)})
        else:
            text = v_chunks.read_page(h["path"])
            material.append({"label": h["title"], "text": store.parse_page(text)[1], "kind": "page",
                             "path": h["path"], "title": h["title"]})
    return material


# ------------------------------------------------------------------ write + verify


def _section(markdown: str, heading: str) -> str:
    """The body of `## heading` (up to the next `## `), or "" when it is missing."""
    m = re.search(rf"^##\s+{heading}\s*$(.*?)(?=^##\s|\Z)", markdown, re.M | re.S | re.I)
    return m.group(1).strip() if m else ""


def write(budget: Budget, question: str, subs: list, material: list) -> tuple:
    """``(title, findings)``: one `long` completion over what was read."""
    blocks = [untrusted(m["text"], m["label"]) for m in material]
    blocks = recall.fit(blocks, llm.CARD_CONTEXT_TOKENS, WRITE_PROMPT)
    user = (f"Question: {question}\nSub-questions: " + "; ".join(s["q"] for s in subs)
            + "\n\nMaterial:\n" + "\n\n".join(blocks))
    reply = _complete(budget, "research_write", WRITE_PROMPT, user, llm.CARD_MAX_TOKENS)
    if not reply:
        raise ValueError("the model wrote nothing")
    findings = _section(reply, "Findings")
    if not findings:                          # no heading: the reply minus its title and any other section
        findings = re.split(r"^##\s", re.sub(r"^#\s.*$", "", reply, count=1, flags=re.M), flags=re.M)[0].strip()
    return llm.card_title(reply, question), findings


_QUOTES = str.maketrans({"“": '"', "”": '"', "„": '"', "«": '"', "»": '"', "‘": "'", "’": "'", "‚": "'"})
#: a bullet's double-quoted spans (empty ones too: they are checked, and fail)
QUOTE = re.compile(r'"([^"\n]*)"')
#: a single-quoted span (never checked: it goes with its citation)
SINGLE = re.compile(r"(?<!\w)'([^'\n]{2,})'(?!\w)")
#: a trailing citation: ` — Title · part 3/12` (an em / en dash or a spaced hyphen, then the source)
CITATION = re.compile(r'\s+(?:[—–]|--?)\s*(?P<ref>[^"\u2014\u2013\n]+?)[\s.;,]*$')
PART = re.compile(r"^(?P<title>.+?) · part (?P<n>\d+)/(?P<total>\d+)$")
MIN_QUOTE = 2


def normalise(text: str) -> str:
    """Curly quotes straightened, `...` as `…`, whitespace collapsed (the verifier's only leeway)."""
    return _flat(str(text or "").translate(_QUOTES).replace("...", "…"))


def _trimmed(quote: str) -> str:
    """The quote without a leading / trailing ellipsis (the only ellipsis allowed)."""
    return normalise(quote).strip().strip("…").strip()


def quoted_in(quote: str, text: str) -> bool:
    """Is `quote` — whitespace / quote-normalised, an outer ellipsis trimmed — a substring of `text`?
    A quote with an inner ellipsis never is (its parts could come from anywhere)."""
    part = _trimmed(quote)
    return len(part) >= MIN_QUOTE and "…" not in part and part in normalise(text)


def _resolve(ref: str, material: list) -> tuple:
    """``(material, cited part n or None, label)`` for a citation; material None when it names nothing read
    (a chunk citation names a part read only when that part was the one asked for)."""
    ref = normalise(ref)
    m = PART.match(ref)
    if m:
        title, n = m.group("title"), int(m.group("n"))
        hit = next((x for x in material if x["kind"] == "chunk" and normalise(x["title"]) == title and x["n"] == n), None)
        return hit, n, ref
    hit = next((x for x in material if x["kind"] == "page" and normalise(x["title"]) == ref), None)
    return hit, None, ref


def _check(quote: str, ref: str, material: list) -> tuple:
    """``(citation to keep, None)`` when the quote verifies against the part it cites, else ``(None, why)``.
    A quote found in a neighbour's text that the cited read returned gets that neighbour's citation."""
    trimmed = _trimmed(quote)
    if "…" in trimmed:
        return None, "contains an ellipsis"
    if len(trimmed) < MIN_QUOTE:
        return None, "no quote to check"
    if not ref:
        return None, "no source cited"
    source, n, label = _resolve(ref, material)
    if source is None or not quoted_in(quote, source["text"]):
        return None, f"not in {label}"
    if n is None:
        return label, None
    parts = source.get("parts") or {n: source["text"]}
    for k in sorted(parts, key=lambda k: abs(k - n)):       # the cited part first, then its neighbours
        if quoted_in(quote, parts[k]):
            return label if k == n else f"{source['title']} · part {k}/{source['total']}", None
    return label, None                           # spans the read text's boundary between two parts


def _split(line: str) -> tuple:
    """``(body, quotes, ref)`` of a quote-straightened Findings line: the claim and its quote without the
    citation, every double-quoted span, and the cited source ("" when none). The citation's source text holds
    no `"`, so it follows the last quote."""
    cite = CITATION.search(line)
    return (line[:cite.start()] if cite else line), QUOTE.findall(line), (cite.group("ref").strip() if cite else "")


def _shape(line: str, quotes: list, ref: str, material: list) -> str:
    """How a Findings line is checked: `claim` (nothing to check), `blockquote`, `bare` (a citation of
    something read but no quote), `many` (two or more quotes) or `one`."""
    if line.lstrip().startswith(">"):
        return "blockquote"
    if not quotes:
        return "bare" if ref and (PART.match(normalise(ref)) or _resolve(ref, material)[0]) else "claim"
    return "many" if len(quotes) > 1 else "one"


#: a connective left hanging where a quote was cut out of a claim
_DANGLING = re.compile(r"\s+(?:and|or|but|plus|vs\.?|versus|&)$", re.I)


def _claim(body: str) -> str:
    """The claim without its quote marks and quoted spans: runs of spaces collapsed and a dangling
    connective or colon trimmed (`- Both: "a" and "b"` is `- Both`)."""
    text = re.sub(r"(?<=\S) {2,}", " ", SINGLE.sub("", QUOTE.sub("", body)))
    while True:
        trimmed = _DANGLING.sub("", text.rstrip(" \u2014\u2013-:,;"))
        if trimmed == text:
            return text
        text = trimmed


def verify(findings: str, material: list) -> tuple:
    """``(findings, verified, total, dropped)``. Every Findings line carrying a quote or a citation counts,
    and so does any `>` blockquote: it must hold exactly one double-quoted span (≥ 2 chars, no inner
    ellipsis) and cite a part this run read whose text holds it. Otherwise its quote and citation go (the
    claim stays; a line left empty, or a blockquote, goes whole) and `{quote, ref, why}` lands in `dropped`."""
    lines, verified, total, dropped = [], 0, 0, []
    for raw in findings.splitlines():
        line = raw.translate(_QUOTES)
        body, quotes, ref = _split(line)
        shape = _shape(line, quotes, ref, material)
        if shape == "claim":
            lines.append(raw)
            continue
        total += 1
        if shape == "one":
            keep, why = _check(quotes[0], ref, material)
        else:
            keep, why = None, "more than one quote" if shape == "many" else "no quote to check"
        if keep is not None:
            verified += 1
            lines.append(f"{body.rstrip()} \u2014 {keep}")
            continue
        claim = _claim(body)
        singles = SINGLE.findall(body) if not quotes else []
        dropped.append({"quote": " / ".join(quotes or singles) or claim.lstrip("> -*").strip(),
                        "ref": normalise(ref), "why": why})
        if shape != "blockquote" and claim.strip(" -*\u2014\u2013:,;."):
            lines.append(claim)
    return "\n".join(lines).strip(), verified, total, dropped


# ------------------------------------------------------------------ the run


def saved_markdown(markdown: str) -> str:
    """What Save to wiki files from a research card: the title, `## Findings` and `## Sources` — never the
    `### Dropped` list (a fabricated quote must not enter the vault)."""
    return re.split(r"\n+### Dropped\n", markdown, maxsplit=1)[0].rstrip() + "\n"


def _markdown(title: str, findings: str, sources: list, dropped: list) -> str:
    out = f"# {title}\n\n## Findings\n{findings or NOTHING_FOUND}\n\n## Sources\n"
    out += "\n".join(f"- {s}" for s in sources) if sources else "(none)"
    if dropped:
        out += "\n\n### Dropped\n" + "\n".join(f'- "{d["quote"]}" — {d["why"]}' for d in dropped)
    return out


def _cited_chunks(findings: str, material: list) -> list:
    """``(source, title, n)`` for every chunk part the verified bullets cite (a neighbour part re-cited by
    `_check` is one the read returned but `material` does not list)."""
    out = []
    for line in findings.splitlines():
        m = PART.match(normalise(_split(line.translate(_QUOTES))[2]))
        read = m and next((x for x in material if x["kind"] == "chunk" and normalise(x["title"]) == m.group("title")), None)
        if read:
            out.append((read["source"], read["title"], int(m.group("n"))))
    return out


def _sources(material: list, findings: str = "") -> list:
    """The `## Sources` lines: `card_sources` over the chunks read and the parts the findings cite, then the
    page titles."""
    chunks = [(m["source"], m["title"], m["n"]) for m in material if m["kind"] == "chunk"]
    chunks += [c for c in _cited_chunks(findings, material) if c not in chunks]
    return llm.card_sources(chunks, [m["title"] for m in material if m["kind"] == "page"])


def _label(text: str) -> str:
    text = _flat(text)
    return text if len(text) <= LABEL_CHARS else text[:LABEL_CHARS - 1] + "…"


def run(question: str, origin: str = "typed", notify=None, request: str = "") -> dict:
    """Research `question` (docs/54-vault-map-research.md) and return its card `{id, markdown, sources,
    title}` + run fields, `reason`, `dropped` and `research` (the run record). The card is issued
    `running` at once and `notify(card, new)` is called with it then and after every finished step;
    a failure ends the run failed (the card keeps that status) and re-raises. `request` is the user's turn
    as the conversation records it (default "research <question>")."""
    notify = notify or (lambda card, new=False: None)
    question = _flat(question)
    flow = runs.Run("research")
    rec = {"run_id": flow.run_id, "question": question, "origin": origin, "status": "running", "reason": "",
           "steps": [], "subquestions": [], "hits": [], "reads": [], "calls": 0, "card_id": ""}
    card = llm.issue_card({"id": llm.next_card_id(), "markdown": "", "sources": [],
                           "title": runs.WORKFLOWS["research"], "reason": "", "dropped": [], "research": rec,
                           **flow.card([runs.OPEN], [{"label": "Plan", "status": "running"}], fold=False)})
    rec["card_id"] = card["id"]
    budget, stage = Budget(), "Plan"

    def done(kind: str, t0: float, detail: str) -> None:
        rec["steps"].append({"kind": kind, "ms": round((time.monotonic() - t0) * 1000), "detail": detail})
        rec["calls"] = budget.calls
        card.update(flow.card([runs.OPEN], fold=False))
        notify(card)

    try:
        notify(card, True)                        # inside: a failing send still ends the run
        t0 = time.monotonic()
        with trace.span("step", name="plan") as span:
            subs = parse_plan(_complete(budget, "research_plan", PLAN_PROMPT, question, PLAN_TOKENS), question)
            span.update(count=len(subs))
        rec["subquestions"] = [s["q"] for s in subs]
        flow.step("Plan", "ok", f"{len(subs)} sub-question{'s' if len(subs) != 1 else ''}")
        done("plan", t0, f"{len(subs)} sub-questions")

        stage, t0 = "Search", time.monotonic()
        hits = search(subs)
        rec["hits"] = hits
        rows = len(flow.steps)
        for i, sub in enumerate(subs):
            n = sum(h["q"] == i for h in hits)
            flow.step(_label(sub["q"]), "ok", f"searched · {n} hit{'s' if n != 1 else ''}")
        done("search", t0, f"{len(hits)} hits")

        stage, t0 = "Judge", time.monotonic()
        with trace.span("step", name="judge") as span:
            judge(budget, question, subs, hits)
            span.update(count=sum(h["keep"] for h in hits))
        for i in range(len(subs)):
            mine = [h for h in hits if h["q"] == i]
            flow.steps[rows + i]["detail"] = (f"searched · {len(mine)} hit{'s' if len(mine) != 1 else ''} · "
                                              f"kept {sum(h['keep'] for h in mine)}")
        done("judge", t0, f"kept {sum(h['keep'] for h in hits)} of {len(hits)}")

        stage, t0 = "Read", time.monotonic()
        with trace.span("step", name="read") as span:
            material = read(budget, hits)
            span.update(count=len(material), sources=[{"title": m["title"], "part": str(m.get("n") or "")}
                                                      for m in material])
        rec["reads"] = [{"source": m.get("source") or m.get("path"), "part": m.get("n")} for m in material]
        flow.step("Read", "ok", f"{len(material)} part{'s' if len(material) != 1 else ''}")
        done("read", t0, f"{len(material)} parts")

        if budget.reason:                         # the governor cut the run: a step of its own
            code = budget.reason.replace(": ", "-")
            with trace.span("step", name="budget", detail=budget.reason, reason=code) as span:
                span["status"] = "skipped"
            flow.step("Budget", "skipped", budget.reason)

        title, findings, verified, total, dropped = question, "", 0, 0, []
        if material:
            stage, t0 = "Wrote", time.monotonic()
            with trace.span("step", name="write"):
                title, findings = write(budget, question, subs, material)
            flow.step("Wrote", "ok")
            done("write", t0, "")

            stage, t0 = "Verified", time.monotonic()
            with trace.span("step", name="verify") as span:
                findings, verified, total, dropped = verify(findings, material)
                span.update(count=verified, skipped=len(dropped))
            flow.step("Verified", "ok" if verified else "skipped", f"{verified}/{total} quotes")
            done("verify", t0, f"{verified}/{total} quotes")

        status = "nothing" if not verified else "partial" if budget.reason else "done"
        reason = "nothing" if status == "nothing" else budget.reason or "ok"
        sources = _sources(material, findings)
        markdown = _markdown(title, findings, sources, dropped)
        rec.update(status=status, reason=reason, calls=budget.calls)
        flow.end(status, reason.replace(": ", "-"))
        card.update(markdown=markdown, sources=sources, title=llm.card_title(markdown, question), reason=reason,
                    dropped=dropped, **flow.card([runs.OPEN, runs.COPY], fold=False))
        llm.file_research(card, request or f"research {question}")
        return card
    except Exception as e:
        flow.fail(stage, runs.failure_reason(e) if not isinstance(e, ValueError) else "failed-model")
        rec.update(status="failed", reason=flow.steps[-1].get("detail", ""), calls=budget.calls)
        card.update(flow.card([runs.OPEN], fold=False))
        notify(card)
        raise
