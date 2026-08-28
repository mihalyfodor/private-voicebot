"""Backdrop catalogue. Images are fetched by scripts/fetch_assets.sh into static/backdrops/ (gitignored)."""
# key → {name, file, credit, url, license}
PIXABAY = "Pixabay Content License"
BACKDROPS = {
    "none": {"name": "None", "file": None, "credit": "", "url": None, "license": ""},
    "meeting": {"name": "Meeting room", "file": "meeting.jpg", "credit": "Backdrop: Juliez4 / Pixabay", "license": PIXABAY,
                "url": "https://cdn.pixabay.com/photo/2024/07/28/15/11/ai-generated-8927764_1280.jpg"},
    "office": {"name": "Home office", "file": "office.jpg", "credit": "Backdrop: Pixabay", "license": PIXABAY,
               "url": "https://cdn.pixabay.com/photo/2024/02/04/01/58/ai-generated-8551386_1280.jpg"},
    "corner": {"name": "Corner office", "file": "corner.jpg", "credit": "Backdrop: Pixabay", "license": PIXABAY,
               "url": "https://cdn.pixabay.com/photo/2024/02/10/22/25/ai-generated-8565636_1280.jpg"},
    "night": {"name": "Anime room, night", "file": "night.jpg", "credit": "Backdrop: vandesart / Pixabay", "license": PIXABAY,
              "url": "https://cdn.pixabay.com/photo/2024/05/26/10/52/anime-8788530_1280.jpg"},
    "library": {"name": "Library", "file": "library.jpg", "credit": "Backdrop: Pixabay", "license": PIXABAY,
                "url": "https://cdn.pixabay.com/photo/2023/04/15/01/07/ai-generated-7926729_1280.jpg"},
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
