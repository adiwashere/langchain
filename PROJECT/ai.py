from langchain_huggingface import ChatHuggingFace, HuggingFaceEndpoint
from dotenv import load_dotenv
from langchain_core.messages import HumanMessage, AIMessage
import requests

from langchain_community.tools import DuckDuckGoSearchRun
from langgraph.prebuilt import create_react_agent
from langchain_core.prompts import ChatPromptTemplate

search_tool = DuckDuckGoSearchRun()
result = search_tool.run("top news in india")

load_dotenv()

llm = HuggingFaceEndpoint(
    repo_id="google/gemma-2-9b-it",
    task="text-generation",
    temperature=0.1
)
model = ChatHuggingFace(llm=llm)    


agent = create_react_agent(
    model,
    tools=[search_tool]
)

result = agent.invoke(
    {"messages": [("user", "top news in india")]}
)

print(result)