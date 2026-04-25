from datetime import datetime

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
