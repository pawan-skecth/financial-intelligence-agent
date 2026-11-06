import json
import math

import yfinance as yf
from dotenv import load_dotenv
from langchain_groq import ChatGroq

load_dotenv()

llm = ChatGroq(
    model="openai/gpt-oss-20b",
    temperature=0,
    max_retries=2,
    timeout=30,
)


def fundamental_analysis(symbol):
    try:
        # ── Step 1: Normalize symbol ──
        symbol = symbol.strip().upper()
        if not symbol:
            return {
                "error": "Stock symbol cannot be empty",
                "raw_metrics": None,
                "llm_analysis": None,
            }

        # ── Step 2: Fetch fundamental data ──
        stock = yf.Ticker(symbol)
        info = stock.get_info()

        raw_metrics = {
            "current_price": info.get("currentPrice"),
            "pe_ratio": info.get("trailingPE"),
            "eps": info.get("trailingEps"),
            "market_cap": info.get("marketCap"),
            "book_value": info.get("bookValue"),
            "dividend_yield": info.get("dividendYield"),
            "52_week_high": info.get("fiftyTwoWeekHigh"),
            "52_week_low": info.get("fiftyTwoWeekLow"),
        }

        # ── Step 3: Sanitize NaN / Inf / non-numeric ──
        for key, value in raw_metrics.items():
            if value is not None:
                try:
                    number = float(value)
                    raw_metrics[key] = number if math.isfinite(number) else None
                except (TypeError, ValueError):
                    raw_metrics[key] = None

        # ── Step 4: Availability check ──
        available_metrics = {k: v for k, v in raw_metrics.items() if v is not None}
        if not available_metrics:
            return {
                "error": f"No fundamental data available for {symbol}",
                "raw_metrics": raw_metrics,
                "llm_analysis": None,
            }

        # ── Step 5: Compute price-to-book (P/B) explicitly ──
        price = raw_metrics.get("current_price")
        book = raw_metrics.get("book_value")

        if price is not None and book is not None and book > 0:
            pb_ratio = price / book
            if pb_ratio > 1.0:
                pb_position = (
                    f"PREMIUM — price is {pb_ratio:.2f}× book value. "
                    f"Market values the company ABOVE its accounting net assets."
                )
            elif pb_ratio < 1.0:
                pb_position = (
                    f"DISCOUNT — price is {pb_ratio:.2f}× book value. "
                    f"Market values the company BELOW its accounting net assets."
                )
            else:
                pb_position = (
                    f"AT PAR — price equals book value (1.00×)."
                )
            pb_line = f"{pb_ratio:.2f}× (trading at {pb_position})"
        else:
            pb_ratio = None
            pb_line = "Unavailable (missing current_price or book_value)"

        raw_metrics["pb_ratio"] = pb_ratio

        # ── Step 6: Dividend yield note ──
        dividend_yield = raw_metrics["dividend_yield"]
        if dividend_yield is not None:
            dividend_yield_note = (
                f"The retrieved dividend yield is {dividend_yield}% — "
                "already expressed as a percentage. Do NOT multiply or divide by 100."
            )
        else:
            dividend_yield_note = (
                "Dividend yield is unavailable. Do not assume a value."
            )

        # ── Step 7: Build prompt with EXPLICIT price-vs-book guidance ──
        prompt = f"""
You are an experienced fundamental stock analyst.

Analyse the following retrieved financial metrics for {symbol}.

RAW FUNDAMENTAL METRICS:
{json.dumps(raw_metrics, indent=2)}

DIVIDEND YIELD INFORMATION:
{dividend_yield_note}

PRICE-TO-BOOK RATIO (already computed for you):
{pb_line}

INSTRUCTIONS:

1. Analyse the P/E ratio and explain valuation.
2. Analyse EPS and profitability.
3. Explain market capitalisation and company size.
4. Analyse book value using the PRE-COMPUTED P/B ratio above.
5. Explain dividend yield.
6. Analyse the 52-week high and low versus current price.
7. Identify fundamental strengths.
8. Identify fundamental weaknesses and risks.
9. Give an overall fundamental assessment:
   Strong / Moderate / Weak.

IMPORTANT RULES — READ CAREFULLY:

- Use the supplied metrics as the source of facts.
- Do not invent missing financial values.
- Dividend yield is already expressed as a percentage.
- Clearly identify unavailable information.

- BOOK VALUE vs PRICE — CRITICAL RULE:
  * The P/B ratio is ALREADY COMPUTED for you above.
  * When describing book value vs price, use the EXACT phrase
    "PREMIUM to book" if P/B > 1, or "DISCOUNT to book" if P/B < 1.
  * NEVER invert this comparison. If P/B > 1, price is ABOVE book,
    NOT below. Do NOT say "market values at discount to book"
    when P/B is above 1.
  * If the P/B ratio is unavailable, explicitly say so — do NOT guess.

- Do not declare a stock undervalued or overvalued based only on one metric.
- Do not make a BUY, HOLD or SELL recommendation.
- Do not assume positive EPS alone proves strong profitability.
- Do not assume book value alone proves financial strength.
- Mention when peer comparisons or historical financial
  information are needed.
- Separate factual observations from your interpretations.
- Explain uncertainty wherever available data is limited.

RESPONSE FORMAT:

1. Fundamental Metrics Overview
2. Valuation Analysis
3. Profitability Analysis
4. Book Value Analysis
5. Dividend Analysis
6. 52-Week Range Analysis
7. Fundamental Strengths
8. Fundamental Weaknesses and Risks
9. Overall Fundamental Assessment

Return a clear and detailed analysis.
"""

        # ── Step 8: Generate analysis ──
        response = llm.invoke(prompt)

        return {
            "raw_metrics": raw_metrics,
            "llm_analysis": response.content,
        }

    except Exception as e:
        return {
            "error": f"Fundamental Analysis Error: {str(e)}",
            "raw_metrics": None,
            "llm_analysis": None,
        }