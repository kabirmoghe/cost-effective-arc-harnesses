"""Reflective Orchestrator (B8) — single agentic loop over a wider tool surface.

Separate from `pipeline/` so the directed-pipeline (B7) architecture stays
pristine and the orchestrator can evolve independently. Orchestrator reuses
pipeline primitives (explorers, code execution, DB client, selection) via
library imports; the dependency is unidirectional (orchestrator → pipeline,
never the reverse).
"""
