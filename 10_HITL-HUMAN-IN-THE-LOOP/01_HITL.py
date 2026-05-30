from langchain_openai import ChatOpenAI
from langgraph.graph import StateGraph, START, END, add_messages
from langgraph.prebuilt.tool_node import tools_condition, ToolNode
from langchain_core.tools import tool
from typing import Annotated, TypedDict
from langchain_core.messages import BaseMessage, HumanMessage, AIMessage, SystemMessage
from pydantic import BaseModel
from langgraph.types import interrupt, Command
from langgraph.checkpoint.memory import MemorySaver
from dotenv import load_dotenv

load_dotenv()
llm = ChatOpenAI(model="gpt-4o-mini")


class ChatState(TypedDict):
    messages: Annotated[list[BaseMessage], add_messages]


def chat_node(state: ChatState):
    decision = interrupt(
        {
            "type": "approval",
            "reason": "Model is about to answer a user question",
            "question": state["messages"][-1].content,
            "instruction": "Approve this question? Yes/No",
        }
    )

    if decision["approved"] == "no":
        return {"messages": AIMessage(content="Not Approved !")}

    else:
        response = llm.invoke(state["messages"])
        return {"messages": [response]}


checkpointer = MemorySaver()
config = {"configurable": {"thread_id": 123}}
graph = StateGraph(ChatState)


graph.add_node("chat_node", chat_node)

graph.add_edge(START, "chat_node")
graph.add_edge("chat_node", END)

workflow = graph.compile(checkpointer=checkpointer)
workflow
while True:
    user_query = input("👉 ")
    if user_query == "exit" or user_query == "quit":
        break

    response = workflow.invoke(
        {"messages": [HumanMessage(content=user_query)]}, config=config
    )
    # print(response)
    if "__interrupt__" in response:
        interrupt_data = response["__interrupt__"][0].value
        
        approval = input(interrupt_data["instruction"] + " ")
        final_response = workflow.invoke(
            Command(resume={"approved": approval}), config=config
        )
        print(final_response["messages"][-1].content)

    else:
        print(response["messages"][-1].content)
