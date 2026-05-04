# Copy this file to triage_rules.py (gitignored) and add your personal rules.
#
# Each rule:
#   sender_contains — case-insensitive substring match on the From header
#   if              — lambda receiving the LLM result dict, returns bool
#   then            — fields to override (category, urgency, or both)

RULES = [
    # Force Action/High for event emails from a specific sender
    # {
    #     "sender_contains": "school name",
    #     "if": lambda r: r.get("event_detected"),
    #     "then": {"category": "Action", "urgency": "High"},
    # },
]
