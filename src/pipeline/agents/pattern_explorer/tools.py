TOOL_DEFINITIONS = [
    {
        "type": "function",
        "function": {
            "name": "think",
            "description": "Record a reasoning step. Use this to analyze examples, form hypotheses, cross-check patterns, and note contradictions.",
            "parameters": {
                "type": "object",
                "properties": {
                    "thought": {
                        "type": "string",
                        "description": "Your reasoning or analysis.",
                    }
                },
                "required": ["thought"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "note_pattern",
            "description": "Record a candidate transformation pattern that you've verified against all example pairs.",
            "parameters": {
                "type": "object",
                "properties": {
                    "pattern": {
                        "type": "string",
                        "description": "A precise description of the transformation pattern.",
                    }
                },
                "required": ["pattern"],
            },
        },
    },
]
