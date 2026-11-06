import re

import streamlit as st
import yfinance as yf
import plotly.graph_objects as go

from workflow.graph import app
from utils.supabase_memory import get_memory


# ============================================================
# PAGE CONFIG
# ============================================================
st.set_page_config(
    page_title="Stock AI Agent",
    layout="wide",
)


# ============================================================
# HELPERS
# ============================================================
def _extract_decision(text: str) -> str:
    """Robust decision extraction — handles **BUY**, casing, whitespace."""
    m = re.search(r"DECISION\s*:?\s*\**\s*(BUY|HOLD|SELL)\b", text.upper())
    return m.group(1) if m else "HOLD"


def _extract_confidence(text: str):
    """Extract integer confidence score from strategy text."""
    m = re.search(r"CONFIDENCE\s*SCORE\s*:?\s*\**\s*(\d{1,3})", text.upper())
    if m:
        try:
            return int(m.group(1))
        except ValueError:
            return None
    return None


def _extract_badge(text: str) -> dict:
    """Parse SENTIMENT_BADGE line."""
    result = {"label": None, "confidence": None}
    m = re.search(
        r"SENTIMENT_BADGE\s*:?\s*\**\s*(POSITIVE|NEGATIVE|NEUTRAL)\s*/\s*\**\s*(HIGH|MEDIUM|LOW)",
        text.upper(),
    )
    if m:
        result["label"] = m.group(1).capitalize()
        result["confidence"] = m.group(2).capitalize()
    return result


def _badge_color(label: str) -> str:
    """Return color code for BUY/SELL/HOLD or Positive/Negative/Neutral."""
    if not label:
        return "gray"
    label = label.upper()
    if label in ("BUY", "POSITIVE"):
        return "green"
    if label in ("SELL", "NEGATIVE"):
        return "red"
    return "orange"


# ============================================================
# SESSION STATE
# ============================================================
if "analysis_result" not in st.session_state:
    st.session_state.analysis_result = None

if "last_stock" not in st.session_state:
    st.session_state.last_stock = None


# ============================================================
# HEADER
# ============================================================
st.title("🧠 Multi-Agent Stock AI System")
st.caption("Fundamental · Technical · Risk · News+Sentiment · Strategy · Memory")


# ============================================================
# INPUTS
# ============================================================
col1, col2, col3 = st.columns([2, 1, 1])

with col1:
    stock = st.text_input("Enter Stock Symbol", "TSLA")

with col2:
    timeframe = st.selectbox(
        "📈 Chart Timeframe",
        ["Daily", "Weekly", "Monthly"],
    )

with col3:
    st.write("")
    st.write("")
    analyze_clicked = st.button("🚀 Analyze", use_container_width=True)


# ============================================================
# RUN ANALYSIS
# ============================================================
if analyze_clicked:
    with st.spinner("Running AI Agents... this may take 30-60 seconds"):
        try:
            st.session_state.analysis_result = app.invoke(
                {"stock_symbol": stock}
            )
            st.session_state.last_stock = stock
        except Exception as e:
            st.error(f"Analysis failed: {e}")


# ============================================================
# SHOW RESULTS
# ============================================================
if st.session_state.analysis_result is not None:
    current_stock = st.session_state.last_stock
    result = st.session_state.analysis_result

    # -------- CANDLESTICK CHART --------
    interval, period = "1d", "6mo"
    if timeframe == "Weekly":
        interval, period = "1wk", "1y"
    elif timeframe == "Monthly":
        interval, period = "1mo", "5y"

    try:
        df = yf.download(
            current_stock,
            period=period,
            interval=interval,
            progress=False,
            auto_adjust=False,
            threads=False,
        )

        if not df.empty:
            if hasattr(df.columns, "levels"):
                df.columns = df.columns.get_level_values(0)

            st.subheader(
                f"📈 {current_stock.upper()} Candlestick Chart ({timeframe})"
            )

            df["SMA20"] = df["Close"].rolling(20).mean()

            fig = go.Figure()
            fig.add_trace(
                go.Candlestick(
                    x=df.index,
                    open=df["Open"], high=df["High"],
                    low=df["Low"], close=df["Close"],
                    name="Candlestick",
                )
            )
            fig.add_trace(
                go.Scatter(
                    x=df.index, y=df["SMA20"],
                    mode="lines", name="SMA 20",
                )
            )
            fig.update_layout(
                height=550,
                xaxis_title="Date",
                yaxis_title="Price",
                xaxis_rangeslider_visible=False,
            )
            st.plotly_chart(fig, use_container_width=True)

    except Exception as e:
        st.error(f"Chart Error: {e}")

    # -------- EXTRACT DECISION + CONFIDENCE + SENTIMENT --------
    strategy_text = result.get("strategy", "") or ""
    decision = _extract_decision(strategy_text)
    confidence = _extract_confidence(strategy_text)
    badge = _extract_badge(strategy_text)

    # -------- DECISION & SENTIMENT BADGE ROW --------
    st.subheader("🎯 AI Recommendation")

    b1, b2, b3 = st.columns(3)

    # Decision badge
    with b1:
        if decision == "BUY":
            st.success(f"🟢 **BUY**")
        elif decision == "SELL":
            st.error(f"🔴 **SELL**")
        else:
            st.warning(f"🟡 **HOLD**")

    # Confidence badge
    with b2:
        if confidence is not None:
            st.info(f"📊 **Confidence: {confidence}%**")
        else:
            st.info("📊 **Confidence: N/A**")

    # Sentiment badge
    with b3:
        label = badge["label"] or "N/A"
        conf = badge["confidence"] or ""
        sent_text = f"{label} ({conf})" if conf else label

        if label == "Positive":
            st.success(f"😊 **Sentiment: {sent_text}**")
        elif label == "Negative":
            st.error(f"😟 **Sentiment: {sent_text}**")
        elif label == "Neutral":
            st.warning(f"😐 **Sentiment: {sent_text}**")
        else:
            st.info("😐 **Sentiment: N/A**")

    # -------- KEY CATALYST (from news_data) --------
    news_data = result.get("news_data") or {}
    sentiment_dict = news_data.get("sentiment") or {}
    key_catalyst = sentiment_dict.get("key_catalyst")

    if key_catalyst:
        st.info(f"🔑 **Key Catalyst:** {key_catalyst}")

    # -------- FULL STRATEGY TEXT --------
    st.subheader("📄 Full Strategy Analysis")
    st.markdown(strategy_text)

    # -------- PAST MEMORY --------
    st.subheader("🧠 Past Decisions (Memory)")

    try:
        history = get_memory(current_stock)
    except Exception as e:
        history = None
        st.error(f"Memory read failed: {e}")

    if history and isinstance(history, list):
        for h in history:
            if isinstance(h, dict):
                ts = str(h.get("timestamp", ""))[:19]
                dec = h.get("decision", "?")
                conf = h.get("confidence", "?")
                emoji = {"BUY": "🟢", "SELL": "🔴", "HOLD": "🟡"}.get(dec, "⚪")
                st.write(f"{emoji} **{dec}** — Confidence: {conf} — {ts}")
            else:
                st.write(h)
    else:
        st.write("No memory found")


# ============================================================
# SIDEBAR — AGENTS LIST (UPDATED)
# ============================================================
st.sidebar.title("🤖 Agents in Pipeline")
st.sidebar.write("""
✔ **Stock Data Tool** — fetches price + metadata  
✔ **Fundamental Agent** — P/E, EPS, book value, etc.  
✔ **Technical Agent** — SMA, RSI, MACD, S/R  
✔ **Risk Agent** — volatility, drawdown, beta  
✔ **News + Sentiment Agent** — headlines + LLM sentiment  
✔ **Strategy Agent** — final decision + confidence  
✔ **Memory Agent** — Supabase persistence  
""")


# ============================================================
# SIDEBAR — INVESTMENT SUMMARY (IMPROVED)
# ============================================================
if st.session_state.analysis_result is not None:
    strategy_text = st.session_state.analysis_result.get("strategy", "") or ""

    st.sidebar.markdown("---")
    st.sidebar.subheader("📋 Quick Summary")

    decision = _extract_decision(strategy_text)
    confidence = _extract_confidence(strategy_text)

    if decision == "BUY":
        st.sidebar.success(f"🟢 BUY ({confidence}%)" if confidence else "🟢 BUY")
    elif decision == "SELL":
        st.sidebar.error(f"🔴 SELL ({confidence}%)" if confidence else "🔴 SELL")
    else:
        st.sidebar.warning(f"🟡 HOLD ({confidence}%)" if confidence else "🟡 HOLD")

    st.sidebar.write("---")
    st.sidebar.write("**Quick Insights:**")

    text_low = strategy_text.lower()
    insights = []

    if "overvalued" in text_low:
        insights.append("⚠️ Overvalued")
    if "undervalued" in text_low:
        insights.append("💎 Undervalued")
    if "bullish" in text_low:
        insights.append("📈 Bullish trend")
    if "bearish" in text_low:
        insights.append("📉 Bearish trend")
    if "low risk" in text_low:
        insights.append("🛡️ Low risk")
    if "high risk" in text_low or "medium-high" in text_low:
        insights.append("⚡ Elevated risk")

    if not insights:
        insights.append("Mixed signals — monitor")

    for i in insights:
        st.sidebar.write(i)

    st.sidebar.write("---")
    st.sidebar.write("**Summary:**")

    if decision == "BUY":
        st.sidebar.write("Multiple positive signals align — favorable setup.")
    elif decision == "SELL":
        st.sidebar.write("Risk and negatives outweigh positives.")
    else:
        st.sidebar.write("Conflicting signals — wait for clarity.")