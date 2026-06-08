"""External-comparator data points for the cost-vs-accuracy Pareto figures.

Numbers sourced from the related-work section of the thesis paper draft
(not the leaderboard CSV — the CSV's Pang/Berman cost values are leaderboard
snapshots, the paper uses author-reported published values).
"""

COMPARATORS = [
    {
        "name": "Greenblatt 2024",
        "short": "Greenblatt",
        "accuracy": 42.0,        # widely-cited public-eval number; cost is the only number from the paper text
        "usd_per_task": 400.0,
        "system_type": "programmatic_synthesis",
    },
    {
        "name": "Berman 2024",
        "short": "Berman '24",
        "accuracy": 53.6,
        "usd_per_task": 29.00,
        "system_type": "evolutionary_python",
    },
    {
        "name": "Berman 2025",
        "short": "Berman '25",
        "accuracy": 79.6,
        "usd_per_task": 8.42,
        "system_type": "evolutionary_nl",
    },
    {
        "name": "Pang 2025",
        "short": "Pang",
        "accuracy": 77.1,
        "usd_per_task": 2.56,
        "system_type": "library_dreamcoder",
    },
    {
        "name": "TRM (Bespoke)",
        "short": "TRM",
        "accuracy": 45.0,
        "usd_per_task": 2.10,
        "system_type": "trained_small_model",
    },
]
