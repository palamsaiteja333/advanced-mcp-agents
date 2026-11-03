import os
from pprint import pprint

from dotenv import load_dotenv
from langchain.agents import create_agent
from langchain.messages import HumanMessage
from langchain.tools import tool
from langchain_aws import ChatBedrockConverse

load_dotenv()

model = ChatBedrockConverse(
    model=os.getenv("AWS_MODEL_ID", "us.amazon.nova-2-lite-v1:0"),
    region_name=os.getenv("AWS_REGION", "us-east-2"),
)


@tool
def square_root(x: float) -> float:
    """Calculate the square root of a number"""
    return x ** 0.5


@tool
def square(x: float) -> float:
    """Calculate the square of a number"""
    return x ** 2


subagent_1 = create_agent(
    model=model,
    tools=[square_root],
)

subagent_2 = create_agent(
    model=model,
    tools=[square],
)


@tool
def call_subagent_1(x: float) -> str:
    """Call subagent 1 in order to calculate the square root of a number"""
    response = subagent_1.invoke({
        "messages": [HumanMessage(content=f"Calculate the square root of {x}")]
    })
    return response["messages"][-1].content


@tool
def call_subagent_2(x: float) -> str:
    """Call subagent 2 in order to calculate the square of a number"""
    response = subagent_2.invoke({
        "messages": [HumanMessage(content=f"Calculate the square of {x}")]
    })
    return response["messages"][-1].content


main_agent = create_agent(
    model=model,
    tools=[call_subagent_1, call_subagent_2],
    system_prompt="You are a helpful assistant who can call subagents to calculate the square root or square of a number.",
)


def main() -> None:
    question = "What is the square root of 456?"
    response = main_agent.invoke({"messages": [HumanMessage(content=question)]})
    pprint(response)


if __name__ == "__main__":
    main()