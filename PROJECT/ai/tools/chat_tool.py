from langchain_core.messages import HumanMessage
from ai.model import model

def normal_chat(user_input):
    response = model.invoke([HumanMessage(content=user_input)])
    return response.content