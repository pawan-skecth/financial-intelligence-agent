
import yfinance as yf
import pandas as pd
import numpy as np
import json

from langchain_groq import ChatGroq
from dotenv import load_dotenv


load_dotenv()

llm = ChatGroq(
    model="openai/gpt-oss-20b",
    temperature=0
)


def technical_analysis(symbol):

    try:

        # STEP 1: DOWNLOAD HISTORICAL DATA

        symbol = symbol.strip().upper()

        if not symbol:
            return {"error": "Stock symbol cannot be empty"}

        df = yf.download(
            tickers=symbol,
            period="1y",
            interval="1d",
            progress=False,
            auto_adjust=True,
            threads=False
        )

        if df is None or df.empty:
            return {"error": "No technical data found"}

        # Handle MultiIndex columns

        if isinstance(df.columns, pd.MultiIndex):
            df.columns = df.columns.get_level_values(0)

        required_columns = ["High", "Low", "Close", "Volume"]

        for column in required_columns:
            if column not in df.columns:
                return {"error": f"{column} column missing"}

        df = df.copy()

        for column in required_columns:
            df[column] = pd.to_numeric(
                df[column], errors="coerce"
            )

        df = df.dropna(subset=["High", "Low", "Close"])

        if len(df) < 50:
            return {"error": "Insufficient historical data"}

        close = df["Close"]
        high = df["High"]
        low = df["Low"]
        volume = df["Volume"]

        latest_price = float(close.iloc[-1])

        # STEP 2: CALCULATE IMPORTANT INDICATORS

        # SMA 20 and SMA 50

        sma_20 = close.rolling(20).mean().iloc[-1]
        sma_50 = close.rolling(50).mean().iloc[-1]

        # RSI 14

        price_change = close.diff()

        gains = price_change.clip(lower=0)
        losses = -price_change.clip(upper=0)

        avg_gain = gains.ewm(
            alpha=1 / 14,
            min_periods=14,
            adjust=False
        ).mean()

        avg_loss = losses.ewm(
            alpha=1 / 14,
            min_periods=14,
            adjust=False
        ).mean()

        rs = avg_gain / avg_loss

        rsi = 100 - (100 / (1 + rs))

        latest_rsi = rsi.iloc[-1]

        # MACD (12, 26, 9)

        ema_12 = close.ewm(
            span=12,
            adjust=False
        ).mean()

        ema_26 = close.ewm(
            span=26,
            adjust=False
        ).mean()

        macd = ema_12 - ema_26

        signal = macd.ewm(
            span=9,
            adjust=False
        ).mean()

        histogram = macd - signal

        # Volume ratio

        average_volume = volume.rolling(20).mean().iloc[-1]

        latest_volume = volume.iloc[-1]

        volume_ratio = None

        if pd.notna(average_volume) and average_volume > 0:
            volume_ratio = float(latest_volume / average_volume)

        # Safely convert values to float

        def safe_float(value):

            if pd.isna(value):
                return None

            value = float(value)

            if not np.isfinite(value):
                return None

            return value

        raw_indicators = {
            "latest_price": latest_price,
            "sma_20": safe_float(sma_20),
            "sma_50": safe_float(sma_50),
            "rsi_14": safe_float(latest_rsi),
            "macd": safe_float(macd.iloc[-1]),
            "macd_signal": safe_float(signal.iloc[-1]),
            "macd_histogram": safe_float(histogram.iloc[-1]),
            "latest_volume": safe_float(latest_volume),
            "average_volume_20": safe_float(average_volume),
            "volume_ratio": volume_ratio
        }

        # STEP 3: BASIC PRICE ACTION ANALYSIS

        recent_df = df.tail(120)

        swing_highs = []
        swing_lows = []

        window = 2

        highs = recent_df["High"].to_numpy()
        lows = recent_df["Low"].to_numpy()

        for i in range(window, len(recent_df) - window):

            # Swing high

            if highs[i] > max(
                highs[i-window:i].tolist()
                + highs[i+1:i+window+1].tolist()
            ):
                swing_highs.append(float(highs[i]))

            # Swing low

            if lows[i] < min(
                lows[i-window:i].tolist()
                + lows[i+1:i+window+1].tolist()
            ):
                swing_lows.append(float(lows[i]))

        # Nearest approximate support and resistance

        support_levels = [
            level for level in swing_lows
            if level < latest_price
        ]

        resistance_levels = [
            level for level in swing_highs
            if level > latest_price
        ]

        support = (
            max(support_levels)
            if support_levels else None
        )

        resistance = (
            min(resistance_levels)
            if resistance_levels else None
        )

        price_action = {
            "recent_swing_highs": swing_highs[-5:],
            "recent_swing_lows": swing_lows[-5:],
            "nearest_support": support,
            "nearest_resistance": resistance
        }

        # STEP 4: SEND RESULTS TO LLM

        prompt = f"""
You are an experienced technical stock market analyst.

Analyse the technical indicators and price action
for stock {symbol}.

CALCULATED TECHNICAL INDICATORS:

{json.dumps(raw_indicators, indent=2)}

CALCULATED PRICE ACTION:

{json.dumps(price_action, indent=2)}

YOUR TASK:

1. Analyse SMA 20 and SMA 50 to understand the trend.
2. Analyse RSI 14 to understand momentum.
3. Analyse MACD to identify bullish or bearish momentum.
4. Analyse volume compared with its 20-day average.
5. Interpret recent swing highs and swing lows.
6. Explain the nearest support and resistance levels.
7. Identify bullish and bearish signals.
8. Provide an overall technical assessment:
   Strong / Moderate / Weak.

IMPORTANT RULES:

- Use only the supplied calculated values.
- Do not invent missing values.
- Do not make a BUY, HOLD or SELL recommendation.
- Do not claim that a stock is oversold or overbought
  without checking the RSI value.
- Do not assume high volume means buying or selling.
  Consider price movement before interpreting volume.
- Do not claim a trend or market structure that
  the supplied data does not support.
- Support and resistance are approximate levels,
  not guaranteed price floors or ceilings.
- Explain conflicting indicators.
- Clearly mention unavailable data.

RESPONSE FORMAT:

1. Trend Analysis
2. RSI and Momentum Analysis
3. MACD Analysis
4. Volume Analysis
5. Price Action Analysis
6. Support and Resistance
7. Bullish Signals
8. Bearish Signals
9. Overall Technical Assessment
"""

        response = llm.invoke(prompt)

        # STEP 5: RETURN STRUCTURED OUTPUT

        return {
            "raw_indicators": raw_indicators,
            "price_action": price_action,
            "llm_analysis": response.content
        }

    except Exception as e:

        return {
            "error": f"Technical Analysis Error: {str(e)}"
        }
