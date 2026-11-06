from typing import TypedDict, Dict, Any


class StockState(TypedDict, total=False):
    stock_symbol: str
    stock_data: Dict[str, Any]
    fundamental_data: Dict[str, Any]
    technical_data: Dict[str, Any]
    risk_data: Dict[str, Any]

    # Merged news + sentiment (dict, not str)
    news_data: Dict[str, Any]

    strategy: str
    analysis: str