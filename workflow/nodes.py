import time

from tools.stock_data import get_stock_data
from agents.fundamental_agent import fundamental_analysis
from agents.technical_agent import technical_analysis
from agents.risk_agent import risk_analysis
from agents.news_agent import analyze_news
from agents.strategy_agent import final_strategy


# 1. STOCK DATA NODE (no LLM — no delay needed)
def stock_data_node(state):
    stock_symbol = state["stock_symbol"]
    data = get_stock_data(stock_symbol)
    print("STOCK DATA:", data)
    return {"stock_data": data}


# 2. FUNDAMENTAL NODE
def fundamental_node(state):
    stock_symbol = state["stock_symbol"]
    data = fundamental_analysis(stock_symbol)
    print("FUNDAMENTAL DATA:", data)
    time.sleep(3)  # throttle for Groq rate limit
    return {"fundamental_data": data}


# 3. TECHNICAL NODE
def technical_node(state):
    stock_symbol = state["stock_symbol"]
    data = technical_analysis(stock_symbol)
    print("TECHNICAL DATA:", data)
    time.sleep(3)  # throttle
    return {"technical_data": data}


# 4. RISK NODE
def risk_node(state):
    stock_symbol = state["stock_symbol"]
    data = risk_analysis(stock_symbol)
    print("RISK DATA:", data)
    time.sleep(3)  # throttle
    return {"risk_data": data}


# 5. NEWS + SENTIMENT NODE (merged)
def news_node(state):
    stock_symbol = state["stock_symbol"]
    data = analyze_news(stock_symbol)
    print("NEWS DATA:", data)
    time.sleep(3)  # throttle
    return {"news_data": data}


# 6. STRATEGY NODE
def strategy_node(state):
    stock_symbol = state["stock_symbol"]
    stock_data = state.get("stock_data")

    if isinstance(stock_data, dict) and stock_data.get("error"):
        return {"strategy": f"HOLD - {stock_data['error']}"}

    combined_data = {
        "stock_data": stock_data,
        "fundamental_data": state.get("fundamental_data"),
        "technical_data": state.get("technical_data"),
        "risk_data": state.get("risk_data"),
        "news_data": state.get("news_data"),
    }

    result = final_strategy(stock_symbol, combined_data)
    return {"strategy": result}