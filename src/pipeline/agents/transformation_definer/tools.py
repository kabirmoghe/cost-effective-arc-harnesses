TOOL_DEFINITIONS = [
    {
        "type": "function",
        "function": {
            "name": "think",
            "description": "Record a reasoning step. Use this to analyze the explorers' findings, compare candidate transformations, identify consensus, and plan your implementation before writing code.",
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
            "name": "define_transformation",
            "description": "Once maximum confidence has been reached, define a transformation that describes the current task's examples.",
            "parameters": {
                "type": "object",
                "properties": {
                    "transformation_summary": {
                        "type": "string",
                        "description": "A brief description of the transformation.",
                    },
                    "reasoning": {
                        "type": "string",
                        "description": "Concise reasoning for why this was the transformation that was defined, including references from PatternExplorer agents, patterns, etc.",
                    },
                    "code": {
                        "type": "string",
                        "description": "Self-contained Python code defining `def transform(grid: list[list[int]]) -> list[list[int]]`. No external imports beyond the standard library. The grid is a 2D list of ints 0-9."
                    }
                },
                "required": ["transformation_summary", "reasoning", "code"],
            },
        },
    }, 
]
