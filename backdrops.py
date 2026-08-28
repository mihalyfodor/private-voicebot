"""Backdrop catalogue. Images are fetched by scripts/fetch_assets.sh into static/backdrops/ (gitignored)."""
# key → {name, file, credit, url, license}
BACKDROPS = {
    "none": {"name": "None", "file": None, "credit": "", "url": None, "license": ""},
}
DEFAULT = "none"


def validate(key: str) -> str:
    key = (key or "").lower()
    if key not in BACKDROPS:
        raise ValueError(f"Unknown backdrop {key!r}; valid: {', '.join(BACKDROPS)}")
    return key


def listing() -> list:
    return [
        {"key": k, "name": v["name"], "file": f"/static/backdrops/{v['file']}" if v["file"] else None, "credit": v["credit"]}
        for k, v in BACKDROPS.items()
    ]
