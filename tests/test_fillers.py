import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
import fillers


def test_pick_never_repeats_consecutively():
    seen = [fillers.pick("get_weather") for _ in range(20)]
    assert all(p in fillers.FILLERS["get_weather"] for p in seen)
    assert all(a != b for a, b in zip(seen, seen[1:]))


def test_unknown_tool_uses_default():
    assert fillers.pick("unknown_tool") in fillers.FILLERS["default"]
