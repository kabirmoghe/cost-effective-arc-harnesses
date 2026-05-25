"""Faithful port of the ARC Prize "Base LLM" harness.

Mirrors the upstream `arcprize/arc-agi-benchmarking` setup verbatim where it
matters: their system prompt, their `json.dumps` grid format, and their pass@2
mechanism (2 attempts per test pair, same context, no variation between
attempts — relying on API non-determinism for diversity). Sampling defaults
(temp 0.0, max_tokens 4024) come from their `models.yml` `deepseek_chat` entry.
"""
