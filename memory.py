import os
from datetime import datetime

SHORTMEM_PATH = os.path.join(os.path.dirname(__file__), "shortmem.txt")


def load(system_prompt: str) -> str:
    if os.path.exists(SHORTMEM_PATH):
        with open(SHORTMEM_PATH, "r") as f:
            content = f.read().strip()
        if content:
            return (
                f"{system_prompt}\n\n"
                "The following is background context about the USER you are speaking with — "
                "it describes them, not you. Use it silently to inform your understanding. "
                f"Never bring it up unless the user does first:\n{content}"
            )
    return system_prompt


def save(session_turns: list, client, model: str):
    existing = ""
    if os.path.exists(SHORTMEM_PATH):
        with open(SHORTMEM_PATH, "r") as f:
            existing = f.read().strip()

    session_text = "\n".join(f"{t['role'].capitalize()}: {t['content']}" for t in session_turns)

    messages = [
        {
            "role": "system",
            "content": (
                "You extract new facts about a user from a conversation transcript. "
                "Compare against existing memory and output only facts that are genuinely new. "
                "One fact per line. No duplicates, no filler, no commentary. "
                "If nothing new was learned, reply with exactly: NOTHING"
            ),
        },
        {
            "role": "user",
            "content": f"Existing memory:\n{existing}\n\nNew session transcript:\n{session_text}",
        },
    ]

    response = client.chat.completions.create(model=model, messages=messages)
    summary = (response.choices[0].message.content or "").strip()

    if not summary or summary.upper() == "NOTHING" or len(summary) < 10:
        print("\n[Nothing new to save]")
        return

    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M")
    with open(SHORTMEM_PATH, "a") as f:
        f.write(f"\n--- {timestamp} ---\n{summary}\n")
    print("\n[Memory saved to shortmem.txt]")
