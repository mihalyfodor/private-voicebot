from datetime import datetime

#: no network at all (docs/33-untrusted-text.md read-only policy)
HOSTS = ()

DEFINITION = {
    "type": "function",
    "function": {
        "name": "get_time",
        "description": "Returns the current local time.",
        "parameters": {"type": "object", "properties": {}, "required": []},
    },
}


def run(args):
    return datetime.now().strftime("%H:%M:%S")
