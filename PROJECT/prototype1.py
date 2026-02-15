import os
import base64
import datetime
from email.mime.text import MIMEText
from dotenv import load_dotenv

from langchain_huggingface import ChatHuggingFace, HuggingFaceEndpoint
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.messages import HumanMessage, AIMessage

# Google APIs
from google_auth_oauthlib.flow import InstalledAppFlow
from googleapiclient.discovery import build
from langchain_community.tools import DuckDuckGoSearchRun

load_dotenv()


llm_endpoint = HuggingFaceEndpoint(
    repo_id="google/gemma-2-9b-it",
    task="text-generation",
    temperature=0.2,
)

model = ChatHuggingFace(llm=llm_endpoint)

chat_history = []
search = DuckDuckGoSearchRun()

intent_prompt = PromptTemplate.from_template("""
Classify the user's message into one word:

CHAT
EMAIL
CALENDAR
NEWS

If the message asks about:
- headlines
- current events
- news
Then classify as NEWS.

Message: {input}

Answer:
""")

intent_chain = intent_prompt | model | StrOutputParser()



calendar_extract_prompt = PromptTemplate.from_template("""
Extract event details from this message.

Message: {input}

Return EXACTLY in this format:

title:
date:
time:
duration_minutes:

Do not add anything else.
""")

email_extract_prompt = PromptTemplate.from_template("""
You are an assistant that writes professional emails.

From the message below:
1. Identify recipient email if mentioned.
2. Generate a clear subject.
3. Write a polite and complete email body.

Rules:
- Do not add placeholders like [Your Name]
- Do not add explanations
- Keep the body under 120 words

Return ONLY in this format:

to:
subject:
body:

Message: {input}
""")

location_extract_prompt = PromptTemplate.from_template("""
Extract location details from this message.

If no location is mentioned, return :  India

Message: {input}

Return only the location name, nothing else.
""")

email_extract_chain = email_extract_prompt | model | StrOutputParser()
calendar_extract_chain = calendar_extract_prompt | model | StrOutputParser()
location_chain = location_extract_prompt | model | StrOutputParser()

def parse_details(text):
    data = {}
    current_key = None

    for line in text.split("\n"):
        line = line.rstrip()

        #If line contains : then it's probably a new key-value pair.
        if ":" in line:
            key, value = line.split(":", 1)
            key = key.strip().lower()
            value = value.strip()

            if key.startswith("to"):
                current_key = "to"
                data[current_key] = value
            elif key.startswith("subject"):
                current_key = "subject"
                data[current_key] = value
            elif key.startswith("body"):
                current_key = "body"
                data[current_key] = value
            elif key.startswith("title"):
                current_key = "title"
                data[current_key] = value
            elif key.startswith("date"):
                current_key = "date"
                data[current_key] = value
            elif key.startswith("time"):
                current_key = "time"
                data[current_key] = value
            elif key.startswith("duration"):
                current_key = "duration_minutes"
                data[current_key] = value
            else:
                current_key = None


        elif current_key:
            data[current_key] += "\n" + line.strip()

    return data


SCOPES = [
    "https://www.googleapis.com/auth/calendar.events",
    "https://www.googleapis.com/auth/gmail.send"
]

def get_services():
    flow = InstalledAppFlow.from_client_secrets_file(
        "credentials.json", SCOPES
    )
    creds = flow.run_local_server(port=0)

    calendar_service = build("calendar", "v3", credentials=creds)
    gmail_service = build("gmail", "v1", credentials=creds)

    return calendar_service, gmail_service


def send_email(gmail_service, to, subject, body):
    message = MIMEText(body)
    message["to"] = to
    message["subject"] = subject

    raw_message = base64.urlsafe_b64encode(
        message.as_bytes()
    ).decode()

    gmail_service.users().messages().send(
        userId="me",
        body={"raw": raw_message}
    ).execute()

def email_tool(user_input):
    details_text = email_extract_chain.invoke({"input": user_input})
    print("\nGenerated Email:\n", details_text)

    data = parse_details(details_text)

    to = data.get("to")
    subject = data.get("subject", "No Subject")
    body = data.get("body", "")

    if not to:
        return "I couldn't find the recipient email."

    calendar_service, gmail_service = get_services()
    send_email(gmail_service, to, subject, body)

    return f"Email sent to {to}"


def create_event(calendar_service, title, date, time, duration):
    start_dt = datetime.datetime.fromisoformat(f"{date}T{time}:00")
    end_dt = start_dt + datetime.timedelta(minutes=int(duration))

    event = {
        "summary": title,
        "start": {
            "dateTime": start_dt.isoformat(),
            "timeZone": "Asia/Kolkata"
        },
        "end": {
            "dateTime": end_dt.isoformat(),
            "timeZone": "Asia/Kolkata"
        },
    }

    calendar_service.events().insert(
        calendarId="primary",
        body=event
    ).execute()

def calendar_tool(user_input):
    details_text = calendar_extract_chain.invoke({"input": user_input})
    print("\nExtracted:\n", details_text)

    data = parse_details(details_text)
    print("Parsed data:", data)

    title = data.get("title", "Meeting")
    date = data.get("date")
    time = data.get("time")
    duration = data.get("duration_minutes", "60")

    # validation
    if not date:
        return "I couldn't understand the date."

    if not time or ":" not in time:
        return "I couldn't understand the time. Please use format HH:MM"

    calendar_service, gmail_service = get_services()
    create_event(calendar_service, title, date, time, duration)

    return f"Event '{title}' scheduled on {date} at {time}"

def news_tool(user_input):
    location = location_chain.invoke({"input": user_input}).strip()

    query = f"Latest news today in {location}"
    results = search.run(query)

    if not results:
        return f"Sorry I couldn't fetch news for {location}."

    summary_prompt = f"""
    Summarize the following news results in 3-4 sentences: \n{results}
    keep it concise and informative and in bullet  points."""
    summary = model.invoke(summary_prompt)
    

    return summary.content



def normal_chat(user_input):
    chat_history.append(HumanMessage(content=user_input))
    response = model.invoke(chat_history)
    chat_history.append(AIMessage(content=response.content))
    return response.content


print("AI Assistant Started (type 'exit' to quit)\n")

while True:
    user_input = input("You: ")

    if user_input.lower() == "exit":
        break

    intent = intent_chain.invoke({"input": user_input}).strip().upper()

    if "EMAIL" in intent:
        reply = email_tool(user_input)
    elif "CALENDAR" in intent:
        reply = calendar_tool(user_input)
    elif "NEWS" in intent:
        reply = news_tool(user_input)
    else:
        reply = normal_chat(user_input)

    print("AI:", reply)