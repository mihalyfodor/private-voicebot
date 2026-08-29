"""docs/index.md is the map of the repo: every test file and PRD must be on it, with the PRD's
current status (the first word of its `Status:` line) in parentheses on every line that names it."""
import os
import re

import pytest

ROOT = os.path.join(os.path.dirname(__file__), "..")
INDEX = open(os.path.join(ROOT, "docs", "index.md")).read()
PRDS = sorted(f for f in os.listdir(os.path.join(ROOT, "docs")) if re.match(r"\d\d[a-z]?-.*\.md$", f))
TESTS = sorted(f for f in os.listdir(os.path.dirname(__file__)) if re.match(r"test_.*\.py$", f))


@pytest.mark.parametrize("name", TESTS)
def test_every_test_file_is_in_the_index(name):
    assert name in INDEX, f"tests/{name} is missing from docs/index.md"


@pytest.mark.parametrize("name", PRDS)
def test_every_prd_is_in_the_index_with_its_status(name):
    head = open(os.path.join(ROOT, "docs", name)).read(400)
    m = re.search(r"^Status:\s*(\w+)", head, re.M)
    assert m, f"docs/{name} has no `Status: <status>` line"
    status = m.group(1)
    lines = [line for line in INDEX.splitlines() if name in line]
    assert lines, f"docs/{name} is missing from docs/index.md"
    for line in lines:
        assert f"({status}" in line, f"docs/index.md says {line.strip()!r} but docs/{name} is `{status}`"
