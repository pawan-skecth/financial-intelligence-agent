
import yfinance as yf
from datetime import datetime, timezone


def get_stock_data(symbol):

    try:

        symbol = symbol.strip().upper()

        if not symbol:
            return {"error": "Stock symbol cannot be empty"}

        stock = yf.Ticker(symbol)

        # Fetch daily historical data
        df = yf.download(
            tickers=symbol,
            period="1mo",
            interval="1d",
            progress=False,
            auto_adjust=True,
            threads=False
        )

        if df is None or df.empty:
            return {"error": f"No market data found for {symbol}"}

        # Fix MultiIndex issue
        if hasattr(df.columns, "levels"):
            df.columns = df.columns.get_level_values(0)

        if "Close" not in df.columns:
            return {"error": "Close price column missing"}

        # Latest available daily price
        latest_close = float(df["Close"].iloc[-1])

        latest_volume = (
            float(df["Volume"].iloc[-1])
            if "Volume" in df.columns
            else None
        )

        # Previous available trading day's close
        previous_close = (
            float(df["Close"].iloc[-2])
            if len(df) >= 2
            else None
        )

        # Price change
        if previous_close is not None:
            price_change = latest_close - previous_close
            price_change_percent = (
                (price_change / previous_close) * 100
                if previous_close != 0 else None
            )
        else:
            price_change = None
            price_change_percent = None

        # Fetch stock metadata
        info = stock.get_info()

        currency = info.get("currency")
        exchange = info.get("exchange")

        # Retrieve the date of the latest daily candle
        latest_date = df.index[-1]

        if hasattr(latest_date, "to_pydatetime"):
            latest_date = latest_date.to_pydatetime()

        if latest_date.tzinfo is None:
            latest_date = latest_date.replace(tzinfo=timezone.utc)

        data_timestamp = latest_date.isoformat()

        # Current retrieval timestamp
        retrieved_at = datetime.now(timezone.utc).isoformat()

        return {
            "symbol": symbol,
            "current_price": round(latest_close, 2),
            "previous_close": previous_close,
            "price_change": price_change,
            "price_change_percent": price_change_percent,
            "volume": latest_volume,
            "currency": currency,
            "exchange": exchange,
            "data_timestamp": data_timestamp,
            "retrieved_at": retrieved_at,
            "price_type": "latest_available_daily_close",
            "data_status": "daily_data"
        }

    except Exception as e:

        return {
            "error": str(e)
        }
