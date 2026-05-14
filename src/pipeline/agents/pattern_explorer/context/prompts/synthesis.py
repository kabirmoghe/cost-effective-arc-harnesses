SYNTHESIS_PROMPT = """\

You have sufficiently explored patterns for this task. Now synthesize your findings into a high-confidence transformation rule. Write a structured analysis in markdown:

1. **Transformation Rule**: A clear, precise description of the overall transformation.
2. **Step-by-step Procedure**: How to apply the transformation to any new input.
3. **Key Observations**: Important details, edge cases, or subtleties.
4. **Confidence**: Low, Medium, High + short description of any remaining uncertainties.

Be concise but complete. This synthesis will be used by a downstream agent to implement the transformation."""
