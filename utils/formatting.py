"""Display formatting helpers for the Streamlit UI."""

from __future__ import annotations


def fmt_moneyline(prob: float) -> str:
    """Convert a win probability (0-1) to American moneyline string."""
    if prob <= 0 or prob >= 1:
        return "N/A"
    if prob >= 0.5:
        ml = -100 * prob / (1 - prob)
        return f"{ml:+.0f}"
    else:
        ml = 100 * (1 - prob) / prob
        return f"+{ml:.0f}"


def fmt_spread(margin: float, team_name: str) -> str:
    """Format a projected margin as a spread string, e.g. 'Duke -6.5'."""
    if abs(margin) < 0.25:
        return "Pick'em"
    sign = "-" if margin > 0 else "+"
    return f"{team_name} {sign}{abs(margin):.1f}"


def fmt_probability(prob: float) -> str:
    """Format a probability as a percentage string."""
    return f"{prob * 100:.1f}%"


def fmt_score(score: float) -> str:
    """Format a projected score to one decimal."""
    return f"{score:.1f}"


def fmt_total(total: float) -> str:
    """Format a projected total."""
    return f"{total:.1f}"


def fmt_edge(model_val: float, market_val: float) -> str:
    """Format the edge between model and market values."""
    edge = model_val - market_val
    if abs(edge) < 0.1:
        return "No Edge"
    return f"{edge:+.1f}"


def confidence_color(label: str) -> str:
    """Return a CSS-friendly color for a confidence label."""
    colors = {
        "Lean": "#f59e0b",
        "Solid": "#3b82f6",
        "Strong Lean": "#10b981",
    }
    return colors.get(label, "#6b7280")
