from langgraph.graph import StateGraph
from workflow.state import StockState

from workflow.nodes import (
    stock_data_node,
    fundamental_node,
    technical_node,
    risk_node,
    news_node,
    strategy_node
)

graph = StateGraph(StockState)

graph.add_node("stock_data", stock_data_node)
graph.add_node("fundamental", fundamental_node)
graph.add_node("technical", technical_node)
graph.add_node("risk", risk_node)
graph.add_node("news", news_node)
graph.add_node("strategy", strategy_node)

graph.set_entry_point("stock_data")

graph.add_edge("stock_data", "fundamental")
graph.add_edge("fundamental", "technical")
graph.add_edge("technical", "risk")
graph.add_edge("risk", "news")
graph.add_edge("news", "strategy")

app = graph.compile()