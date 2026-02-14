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

        # If line does NOT contain ":", it belongs to previous field
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
