from ai.tools.email_tool import email_tool
from ai.tools.calendar_tool import calendar_tool
from ai.tools.news_tool import news_tool
from ai.model import model
from ai.prompts import intent_prompt
from langchain_core.output_parsers import StrOutputParser

intent_chain = intent_prompt | model | StrOutputParser()

def run_assistant(user_input):

    intent = intent_chain.invoke({"input": user_input}).strip().upper()

    if intent == "EMAIL":
        return email_tool(user_input)

    elif intent == "CALENDAR":
        return calendar_tool(user_input)

    elif intent == "NEWS":
        return news_tool(user_input)

    else:
        response = model.invoke(user_input)
        return response.content