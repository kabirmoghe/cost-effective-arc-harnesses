# Baseline

```mermaid
%%{init: {'flowchart': {'nodeSpacing': 30, 'rankSpacing': 60}}}%%
flowchart LR
    FS["Few-shot examples<br>(x₁,x₂...xₙ)"] --> LLM["LLM"]
    LLM --> OUT["output.json"]
```
