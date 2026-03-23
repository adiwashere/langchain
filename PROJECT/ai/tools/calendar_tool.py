from ai.utils.parser import parse_details
from ai.services.calendar_service import create_event
from ai.model import model
from ai.prompts import calendar_extract_prompt
from langchain_core.output_parsers import StrOutputParser

calendar_extract_chain = calendar_extract_prompt | model | StrOutputParser()

def calendar_tool(user_input, session):

    details_text = calendar_extract_chain.invoke({"input": user_input})

    data = parse_details(details_text)

    title = data.get("title", "Meeting")
    date = data.get("date")
    time = data.get("time")
    duration = data.get("duration_minutes", "60")

    if not date:
        return "I couldn't understand the date."

    if not time or ":" not in time:
        return "Time must be HH:MM."

    from ai.services.gmail_service import get_gmail_service

    calendar_service, _ = get_gmail_service()

    create_event(calendar_service, title, date, time, duration)

    return f"Event '{title}' scheduled on {date} at {time}"