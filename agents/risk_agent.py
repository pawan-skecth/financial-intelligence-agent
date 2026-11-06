
import yfinance as yf
import numpy as np
import json

from langchain_groq import ChatGroq
from dotenv import load_dotenv


load_dotenv()

llm = ChatGroq(
    model="openai/gpt-oss-20b",
    temperature=0
)


def risk_analysis(symbol):

    try:

        # STEP 1: DOWNLOAD STOCK DATA

        symbol = symbol.strip().upper()

        if not symbol:
            return {"error": "Stock symbol cannot be empty"}

        stock_df = yf.download(
            tickers=symbol,
            period="1y",
            interval="1d",
            progress=False,
            auto_adjust=True,
            threads=False
        )

        if stock_df is None or stock_df.empty:
            return {"error": "No stock risk data found"}

        if isinstance(stock_df.columns, __import__("pandas").MultiIndex):
            stock_df.columns = stock_df.columns.get_level_values(0)

        if "Close" not in stock_df.columns:
            return {"error": "Stock Close column missing"}

        stock_returns = stock_df["Close"].pct_change().dropna()

        if len(stock_returns) < 30:
            return {"error": "Insufficient stock return data"}

        # STEP 2: CALCULATE VOLATILITY

        daily_volatility = float(stock_returns.std())

        annualized_volatility = float(
            daily_volatility * np.sqrt(252)
        ) * 100

        # STEP 3: CALCULATE MAXIMUM DRAWDOWN

        prices = stock_df["Close"].dropna()

        running_peak = prices.cummax()

        drawdown = (prices - running_peak) / running_peak

        maximum_drawdown = float(drawdown.min() * 100)

        # STEP 4: DOWNLOAD NIFTY 50 DATA FOR BETA

        market_df = yf.download(
            tickers="^NSEI",
            period="1y",
            interval="1d",
            progress=False,
            auto_adjust=True,
            threads=False
        )

        beta = None

        if market_df is not None and not market_df.empty:

            if isinstance(market_df.columns, __import__("pandas").MultiIndex):
                market_df.columns = market_df.columns.get_level_values(0)

            if "Close" in market_df.columns:

                market_returns = (
                    market_df["Close"].pct_change().dropna()
                )

                # Align stock and market returns by date

                aligned_returns = __import__("pandas").concat(
                    [
                        stock_returns.rename("stock"),
                        market_returns.rename("market")
                    ],
                    axis=1,
                    join="inner"
                ).dropna()

                if len(aligned_returns) >= 30:

                    market_variance = aligned_returns["market"].var()

                    if market_variance > 0:

                        beta = float(
                            aligned_returns["stock"].cov(
                                aligned_returns["market"]
                            ) / market_variance
                        )

        # STEP 5: STORE RAW RISK METRICS

        raw_metrics = {
            "daily_volatility": daily_volatility,
            "annualized_volatility_percent": annualized_volatility,
            "maximum_drawdown_percent": maximum_drawdown,
            "beta": beta,
            "historical_return_days": len(stock_returns)
        }

        # STEP 6: SEND METRICS TO LLM

        prompt = f"""
You are an experienced financial risk analyst.

Analyse the following calculated risk metrics for {symbol}.

RAW RISK METRICS:

{json.dumps(raw_metrics, indent=2)}

INSTRUCTIONS:

1. Explain daily volatility.
2. Explain annualized volatility as a percentage.
3. Analyse maximum drawdown and historical downside risk.
4. Analyse beta compared with NIFTY 50.
5. Explain whether the stock has relatively high or low risk.
6. Identify the main risk factors based on the supplied metrics.
7. Give an overall risk assessment:
   Low / Medium / High.

IMPORTANT RULES:

- Use only the supplied calculated metrics.
- Do not invent missing values.
- If beta is unavailable, clearly mention it.
- Do not treat historical risk as a guarantee of future risk.
- Do not confuse beta with total risk.
- Explain that volatility measures price fluctuations.
- Explain that maximum drawdown measures the largest
  historical peak-to-trough percentage fall.
- Do not make a BUY, HOLD or SELL recommendation.
- Do not claim that a stock is safe just because beta is low.

RESPONSE FORMAT:

1. Volatility Analysis
2. Maximum Drawdown Analysis
3. Beta Analysis
4. Risk Factors
5. Overall Risk Assessment
"""

        response = llm.invoke(prompt)

        # STEP 7: RETURN STRUCTURED OUTPUT

        return {
            "raw_metrics": raw_metrics,
            "llm_analysis": response.content
        }

    except Exception as e:

        return {
            "error": f"Risk Analysis Error: {str(e)}"
        }
