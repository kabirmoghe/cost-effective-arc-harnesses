"""Parse LLM responses to extract thoughts and tool calls."""

import json
import re
from typing import List, Dict, Any, Optional
from pydantic import BaseModel, Field


class ToolCall(BaseModel):
    """A parsed tool call from LLM response."""

    name: str = Field(description="Tool name")
    arguments: Dict[str, Any] = Field(description="Tool arguments")
    raw: str = Field(default="", description="Raw text of the tool call")


class ParsedResponse(BaseModel):
    """Parsed LLM response containing thoughts and tool calls."""

    thought: Optional[str] = Field(default=None, description="The reasoning/thought from the LLM")
    tool_calls: List[ToolCall] = Field(default_factory=list, description="Tool calls extracted")
    raw_content: str = Field(default="", description="Raw response content")


def parse_response(response: Any) -> ParsedResponse:
    """Parse an LLM response to extract thoughts and tool calls.

    Handles both:
    1. OpenAI/DeepSeek function calling responses with tool_calls
    2. Text-based responses with <thought> and <tool> XML tags

    Args:
        response: The raw response from the LLM client

    Returns:
        ParsedResponse with extracted thoughts and tool calls
    """
    # Handle OpenAI-style response objects
    if hasattr(response, "choices") and response.choices:
        choice = response.choices[0]
        message = choice.message

        thought = None
        tool_calls = []

        # Extract content (thought)
        if message.content:
            thought = message.content.strip()
            # Try to extract thought from XML tags if present
            thought_match = re.search(r"<thought>(.*?)</thought>", thought, re.DOTALL)
            if thought_match:
                thought = thought_match.group(1).strip()

        # Extract tool calls from function calling
        if hasattr(message, "tool_calls") and message.tool_calls:
            for tc in message.tool_calls:
                try:
                    args = json.loads(tc.function.arguments) if tc.function.arguments else {}
                except json.JSONDecodeError:
                    args = {}

                tool_calls.append(ToolCall(
                    name=tc.function.name,
                    arguments=args,
                    raw=tc.function.arguments or "",
                ))

        return ParsedResponse(
            thought=thought,
            tool_calls=tool_calls,
            raw_content=message.content or "",
        )

    # Handle string/text responses (parse XML-style tags)
    if isinstance(response, str):
        return parse_text_response(response)

    # Handle dict responses
    if isinstance(response, dict):
        content = response.get("content", "")
        if isinstance(content, str):
            return parse_text_response(content)

    return ParsedResponse(raw_content=str(response))


def parse_text_response(text: str) -> ParsedResponse:
    """Parse a text response with XML-style tags."""
    thought = None
    tool_calls = []

    # Extract thought
    thought_match = re.search(r"<thought>(.*?)</thought>", text, re.DOTALL)
    if thought_match:
        thought = thought_match.group(1).strip()

    # Extract tool calls
    tool_pattern = r'<tool\s+name="([^"]+)">(.*?)</tool>'
    for match in re.finditer(tool_pattern, text, re.DOTALL):
        tool_name = match.group(1)
        args_text = match.group(2).strip()

        # Try to parse arguments
        args = parse_tool_arguments(args_text)

        tool_calls.append(ToolCall(
            name=tool_name,
            arguments=args,
            raw=args_text,
        ))

    return ParsedResponse(
        thought=thought,
        tool_calls=tool_calls,
        raw_content=text,
    )


def parse_tool_arguments(args_text: str) -> Dict[str, Any]:
    """Parse tool arguments from text.

    Handles both:
    - JSON format: {"key": "value"}
    - Key=value format: key=value, key2=value2
    """
    args_text = args_text.strip()

    # Try JSON first
    if args_text.startswith("{"):
        try:
            return json.loads(args_text)
        except json.JSONDecodeError:
            pass

    # Try key=value format
    args = {}
    # Match patterns like: key=value or key=[1, 2, 3]
    pattern = r'(\w+)\s*=\s*(\[[^\]]*\]|\{[^}]*\}|"[^"]*"|\'[^\']*\'|[^,\s]+)'
    for match in re.finditer(pattern, args_text):
        key = match.group(1)
        value_str = match.group(2).strip()

        # Parse the value
        try:
            # Try JSON parsing for arrays, objects, strings
            value = json.loads(value_str)
        except json.JSONDecodeError:
            # Keep as string
            value = value_str.strip("'\"")

            # Try numeric conversion
            try:
                if "." in value:
                    value = float(value)
                else:
                    value = int(value)
            except ValueError:
                pass

        args[key] = value

    return args
