from langchain_openai import ChatOpenAI
from dotenv import load_dotenv
from langgraph.graph import StateGraph, START, END
from typing import TypedDict, Annotated
from langgraph.checkpoint.memory import MemorySaver
from langgraph.graph.message import add_messages
from langchain_core.messages import BaseMessage, HumanMessage, AIMessage
from langgraph.prebuilt import ToolNode, tools_condition
from langchain_community.tools import DuckDuckGoSearchRun
from langgraph.types import Command, interrupt
from langgraph.checkpoint.memory import MemorySaver
from langchain_core.tools import tool

load_dotenv()
llm = ChatOpenAI(model="gpt-4o-mini")
# Tools
import requests

search_tool = DuckDuckGoSearchRun(region="us-en")


@tool
def calculator(a: float, b: float, operation: str) -> dict:
    """
    Perform a basic arithmetic operation of two numbers.
    Supported opeartions are : add , sub , mul , div
    """
    if operation == "add":
        result = a + b
    elif operation == "sub":
        result = a - b
    elif operation == "mul":
        result = a * b
    elif operation == "div":
        if b == 0:
            return {"error": "Not allowed"}
        result = a / b
    else:
        return {"error": f"Unsupported Operation {operation}"}
    return {"result": result}


@tool
def get_stock_price(symbol: str) -> dict:
    """
    Fetch latest stock price for a given symbol (e.g. 'AAPL','TSLA')
    using Alpha Vantage api key in the URL.
    """

    decision = interrupt("Permission to use this tool")
    if decision["approved"].lower() == "no":
        return {
            "User denied access. Politely acknowledge and move on."
        }  
    else:
        url = f"https://www.alphavantage.co/query?function=GLOBAL_QUOTE&symbol={symbol}&apikey=9E2OTGJ1SW6VEM8B"
        r = requests.get(url)
        return r.json()


tools = [search_tool, calculator, get_stock_price]
llm_with_tools = llm.bind_tools(tools)
checkpointer = MemorySaver()
# state


class ChatState(TypedDict):
    messages: Annotated[list[BaseMessage], add_messages]


# graph nodes


def chat_node(state: ChatState):
    messages = state["messages"]
    response = llm_with_tools.invoke(messages)

    return {"messages": [response]}


tool_node = ToolNode(tools)
graph = StateGraph(ChatState)
graph.add_node("chat_node", chat_node)
graph.add_node("tools", tool_node)

graph.add_edge(START, "chat_node")
graph.add_conditional_edges("chat_node", tools_condition)
graph.add_edge("tools", "chat_node")

workflow = graph.compile(checkpointer=checkpointer)
config = {"configurable": {"thread_id": "1233"}}

while True:
    user_query = input("You: ")

    if user_query.lower() in ["exit", "quit"]:
        break
    response = workflow.invoke(
        {"messages": [HumanMessage(content=user_query)]}, config=config
    )

    if "__interrupt__" in response:
        interrupt_data = response["__interrupt__"][0].value
        hitl = input(
            f'Should I use {response["messages"][-1].tool_calls[-1]["name"]}? '
        )
        final_response = workflow.invoke(
            Command(resume={"approved": hitl}), config=config
        )
        print(final_response["messages"][-1].content)

    else:
        print(response["messages"][-1].content)
