"""FinSight Dashboard — Main Entry Point.

Streamlit multi-page application with sidebar navigation.

Usage:
    streamlit run src/dashboard/app.py
"""

from __future__ import annotations

import streamlit as st

# ---------------------------------------------------------------------------
# Page config (must be first Streamlit call)
# ---------------------------------------------------------------------------
st.set_page_config(
    page_title="FinSight",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ---------------------------------------------------------------------------
# Custom CSS for professional financial styling
# ---------------------------------------------------------------------------
st.markdown("""
<style>
/* Dark header bar */
header[data-testid="stHeader"] { background: #0e1117; }

/* Metric cards */
div[data-testid="stMetric"] {
    background: #1e1e2f;
    border: 1px solid #2d2d44;
    border-radius: 8px;
    padding: 12px 16px;
}
div[data-testid="stMetric"] label { color: #888; font-size: 12px; }
div[data-testid="stMetric"] [data-testid="stMetricValue"] { color: #e0e0e0; }

/* Green/Red delta colors */
div[data-testid="stMetric"] [data-testid="stMetricDelta"] svg { display: none; }

/* Sidebar styling */
section[data-testid="stSidebar"] {
    background: #0e1117;
    border-right: 1px solid #1e1e2f;
}

/* Tab styling */
button[data-baseweb="tab"] { font-size: 14px; }

/* Tables */
table { font-size: 13px; }
</style>
""", unsafe_allow_html=True)


# ---------------------------------------------------------------------------
# Page registry
# ---------------------------------------------------------------------------
PAGES = {
    "Live Performance": "live_performance",
    "Portfolio Overview": "portfolio",
    "AI Research Assistant": "research",
    "Backtest Lab": "backtest_lab",
    "Risk Dashboard": "risk",
    "Data Explorer": "data_explorer",
}


def main() -> None:
    """Main dashboard entry point."""

    # Sidebar navigation
    with st.sidebar:
        st.markdown("## FinSight")
        st.caption("AI-Powered Investment Research")
        st.divider()

        page = st.radio(
            "Navigation",
            list(PAGES.keys()),
            label_visibility="collapsed",
        )

        st.divider()

        # System status
        st.markdown("##### System Status")
        st.markdown("🟢 Dashboard Online")
        st.markdown("🟡 DB: Offline (demo mode)")
        st.markdown("📊 Data: Synthetic sample")

        st.divider()
        st.caption("v0.4.0 — Phase 3 Complete")

    # Route to selected page
    page_key = PAGES[page]

    if page_key == "live_performance":
        from src.dashboard._pages.live_performance import render
        render()
    elif page_key == "portfolio":
        from src.dashboard._pages.portfolio import render
        render()
    elif page_key == "research":
        from src.dashboard._pages.research import render
        render()
    elif page_key == "backtest_lab":
        from src.dashboard._pages.backtest_lab import render
        render()
    elif page_key == "risk":
        from src.dashboard._pages.risk import render
        render()
    elif page_key == "data_explorer":
        from src.dashboard._pages.data_explorer import render
        render()


if __name__ == "__main__":
    main()
