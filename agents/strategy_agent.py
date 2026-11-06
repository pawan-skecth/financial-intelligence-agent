import re

from langchain_groq import ChatGroq
from dotenv import load_dotenv

from utils.supabase_memory import save_memory, get_memory

load_dotenv()

llm = ChatGroq(
    model="openai/gpt-oss-20b",
    temperature=0,
    max_retries=2,
    timeout=45,
)


DISCLAIMER = (
    "\n\n---\n"
    "*This analysis is for educational purposes only and is not "
    "financial advice. Please consult a SEBI-registered investment "
    "advisor before making investment decisions.*"
)


# ============================================================
# FORMAT HELPERS
# ============================================================

def _format_news(news_data):
    """Return short news summary + structured sentiment block."""
    if not news_data or not isinstance(news_data, dict) or news_data.get("error"):
        return (
            "No recent news available.",
            "Label: Neutral\nConfidence: Low\n"
            "Market Impact Direction: Unclear\nKey Catalyst: (none)",
        )

    full_text = news_data.get("analysis_text") or ""
    short_news = full_text[:600] + ("..." if len(full_text) > 600 else "")

    s = news_data.get("sentiment") or {}
    sent_block = (
        f"Label: {s.get('sentiment_label', 'Neutral')}\n"
        f"Confidence: {s.get('sentiment_confidence', 'Low')}\n"
        f"Market Impact: {s.get('market_impact_direction', 'Unclear')}\n"
        f"Key Catalyst: {s.get('key_catalyst') or '(none)'}"
    )
    return short_news, sent_block


def _format_other(combined_data):
    """Return compact raw metrics only."""
    if not isinstance(combined_data, dict):
        return "No data.", "Unavailable"

    stock = combined_data.get("stock_data") or {}
    price = stock.get("current_price")
    currency = stock.get("currency") or ""

    if price is not None:
        currency_tag = currency if currency else "INR"
        price_line = f"{round(price, 2)} {currency_tag}"
    else:
        price_line = "Unavailable"

    fund = combined_data.get("fundamental_data") or {}
    tech = combined_data.get("technical_data") or {}
    risk = combined_data.get("risk_data") or {}

    compact = {
        "fundamental": fund.get("raw_metrics"),
        "technical": tech.get("raw_indicators"),
        "price_action": tech.get("price_action"),
        "risk": risk.get("raw_metrics"),
    }

    return str(compact), price_line


def _format_memory(history):
    if isinstance(history, list) and history:
        return str(history[:2])
    if not history:
        return "No previous memory."
    return str(history)


# ============================================================
# EXTRACTION HELPERS
# ============================================================

def _extract_decision(text):
    m = re.search(r"DECISION\s*:?\s*\**\s*(BUY|HOLD|SELL)\b", text.upper())
    return m.group(1) if m else "HOLD"


def _extract_confidence(text):
    m = re.search(r"CONFIDENCE\s*SCORE\s*:?\s*\**\s*(\d{1,3})", text.upper())
    if m:
        try:
            return max(0, min(100, int(m.group(1))))
        except ValueError:
            return "N/A"
    return "N/A"


# ============================================================
# MAIN FUNCTION
# ============================================================

def final_strategy(stock_symbol, combined_data):

    if not combined_data:
        return "No data available"

    try:
        history = get_memory(stock_symbol)
    except Exception as e:
        print(f"MEMORY READ FAILED: {e}")
        history = "No previous memory available."

    history_str = _format_memory(history)

    news_summary, sentiment_block = _format_news(
        combined_data.get("news_data") if isinstance(combined_data, dict) else None
    )
    compact_data, price_line = _format_other(combined_data)

    prompt = f"""
You are an expert AI Stock Market Analyst.

Analyze ALL the information below and produce a balanced, professional
final strategy. Cover every agent, but keep each section concise.

════════════════════════════════════════════════
CURRENT PRICE (use this exact value, INCLUDING the currency code):
{price_line}

COMPACT METRICS (ground truth — treat as facts):
{compact_data}

NEWS SUMMARY (context only — treat as interpretation):
{news_summary}

SENTIMENT VERDICT (authoritative — trust this over the news text):
{sentiment_block}

PREVIOUS MEMORY:
{history_str}
════════════════════════════════════════════════

════════════════════════════════════════════════
DECISION RUBRIC — APPLY STRICTLY, DO NOT DEFAULT TO HOLD
════════════════════════════════════════════════

Evaluate these FOUR conditions. Mark each YES / NO / NEUTRAL based on data:

  C1. Fundamental Valuation:
      YES     if EPS is positive AND valuation verdict is
              "Undervalued" or "Fairly Valued".
      NO      if valuation verdict is "Overvalued".
      NEUTRAL if data is unavailable or unclear.

  C2. Technical Trend:
      YES     if price > SMA-20 AND price > SMA-50
              (or RSI > 55 with positive MACD histogram).
      NO      if price < SMA-20 AND price < SMA-50
              (or MACD histogram negative AND RSI < 45).
      NEUTRAL otherwise (mixed signals).

  C3. Sentiment:
      YES     if sentiment_label = "Positive"
              (any confidence, Medium or High preferred).
      NO      if sentiment_label = "Negative"
              (any confidence, Medium or High preferred).
      NEUTRAL if sentiment_label = "Neutral".

  C4. Risk Acceptability:
      YES     if annualized volatility < 30%
              AND max drawdown > -35%.
      NO      if annualized volatility >= 35%
              OR max drawdown <= -40%.
      NEUTRAL otherwise.

════════════════════════════════════════════════
DECISION LOGIC — COUNT CAREFULLY
════════════════════════════════════════════════

Count the number of YES, NO, and NEUTRAL across C1–C4.

  → BUY  if:
      * at least 3 conditions are YES, AND
      * C4 (risk) is NOT NO.

    Valid BUY patterns:
      - 4 YES                                   → BUY
      - 3 YES + 1 NEUTRAL                       → BUY
      - C1=YES, C2=YES, C3=YES, C4=NEUTRAL      → BUY
      - C1=YES, C2=YES, C3=NEUTRAL, C4=YES      → BUY
      - C1=YES, C2=NEUTRAL, C3=YES, C4=YES      → BUY
      - C1=NEUTRAL, C2=YES, C3=YES, C4=YES      → BUY

    Invalid BUY patterns (HOLD instead):
      - Only 2 YES                              → HOLD
      - 3 YES but C4 = NO                       → HOLD

  → SELL if:
      * at least 2 of C1, C2, C3 are NO, AND
      * C4 is NEUTRAL or NO.

    Valid SELL patterns:
      - C1=NO + C2=NO                           → SELL
      - C1=NO + C3=NO                           → SELL
      - C2=NO + C3=NO                           → SELL
      - 3 NO                                    → SELL

    Invalid SELL patterns (HOLD instead):
      - Only 1 NO                               → HOLD
      - 2 NO but C4 = YES (acceptable risk)     → HOLD

  → HOLD if neither BUY nor SELL conditions are met.

CRITICAL:
- Do NOT default to HOLD. Count first, then decide.
- If BUY pattern matches → choose BUY.
- If SELL pattern matches → choose SELL.
- HOLD is ONLY for genuinely mixed cases.

════════════════════════════════════════════════
CONFIDENCE SCORE RUBRIC
════════════════════════════════════════════════

Start at 50.
+10 for each C1–C4 that is YES.
+5  for each C1–C4 that is NEUTRAL.
-10 for each C1–C4 that is NO.
-10 if sentiment_confidence is "Low".
-5  if any metric above is "Unavailable" or "None".
-5  if news_summary was truncated (contains "...").
Cap at 95. Floor at 30.
Write as a single integer.

Example: 4 YES → 50 + 40 = 90.
Example: 3 YES + 1 NEUTRAL → 50 + 30 + 5 = 85.
Example: 2 YES + 1 NEUTRAL + 1 NO → 50 + 20 + 5 - 10 = 65.
Example: 1 YES + 3 NEUTRAL → 50 + 10 + 15 = 75.
Example: 2 NO + 2 NEUTRAL → 50 - 20 + 10 = 40.

════════════════════════════════════════════════
OUTPUT RULES
════════════════════════════════════════════════

1. DECISION: Exactly ONE of BUY / HOLD / SELL. Place at the very top.
2. Current Stock Price: Use CURRENT PRICE above verbatim, WITH currency code.
3. Fundamental Analysis (3-4 lines): P/E, EPS, Market Cap, Book Value,
   Dividend Yield, 52W High/Low, and classification
   (Undervalued / Fairly Valued / Overvalued).
4. Technical Analysis (2-3 lines): trend, RSI/MACD/SMA signal,
   nearest support/resistance.
5. Risk Analysis (2 lines): volatility, drawdown, beta, risk verdict.
6. News & Sentiment Analysis (2-3 lines): sentiment label + confidence,
   main catalyst, counter-balancing factor.
7. Reason (1-2 sentences, max 40 words): synthesize using the C1–C4 counts.
8. Confidence Score: integer per the rubric above.
9. SENTIMENT_BADGE: mirror SENTIMENT VERDICT label and confidence EXACTLY.

STYLE:
- No markdown tables. No bullet spam. Plain prose.
- Total output ~200-300 words.
- Currency symbol: ₹ for INR, $ for USD.

RESPONSE FORMAT (strict):

DECISION: BUY/HOLD/SELL

Current Stock Price: <value> <INR|USD>

Fundamental Analysis:
<3-4 lines>

Technical Analysis:
<2-3 lines>

Risk Analysis:
<2 lines>

News & Sentiment Analysis:
<2-3 lines>

Reason:
<1-2 sentences, max 40 words>

Confidence Score: <0-100>

SENTIMENT_BADGE: <Positive|Negative|Neutral> / <High|Medium|Low>
"""

    try:
        response = llm.invoke(prompt)
        output = (response.content or "").strip()
    except Exception as e:
        return f"Strategy generation failed: {str(e)}"

    decision = _extract_decision(output)
    confidence = _extract_confidence(output)

    try:
        save_memory(stock_symbol, decision, confidence)
    except Exception as e:
        print(f"MEMORY SAVE FAILED: {e}")

    return output + DISCLAIMER