from ai.tools.email_tool import email_tool
from ai.tools.calendar_tool import calendar_tool
from ai.tools.news_tool import news_tool
from ai.tools.chat_tool import normal_chat

from ai.model import model
from ai.prompts import intent_prompt
from langchain_core.output_parsers import StrOutputParser

intent_chain = intent_prompt | model | StrOutputParser()

user_sessions = {}

def handle_request(user_input, session_id="default"):

    if session_id not in user_sessions:
        user_sessions[session_id] = {
            "pending_email": None
        }

    session = user_sessions[session_id]

    intent = intent_chain.invoke({"input": user_input}).strip().upper()

    if intent not in ["CHAT", "EMAIL", "CALENDAR", "NEWS"]:
        intent = "CHAT"

    # ✅ FIXED CONFIRM CHECK
    if session["pending_email"] and user_input.strip().lower() in ["confirm", "yes", "send", "ok"]:
        return email_tool(user_input, session)

    if intent == "EMAIL":
        return email_tool(user_input, session)

    elif intent == "CALENDAR":
        return calendar_tool(user_input)

    elif intent == "NEWS":
        return news_tool(user_input)

    else:
        return normal_chat(user_input)