import os
import base64
import datetime
from email.mime.text import MIMEText
from dotenv import load_dotenv
import re
from langchain_huggingface import ChatHuggingFace, HuggingFaceEndpoint
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.messages import HumanMessage, AIMessage
from langchain_google_genai import ChatGoogleGenerativeAI

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
# model = ChatGoogleGenerativeAI(model="gemini-2.5-flash", temperature=0.2)   

chat_history = []
pending_email = None
search = DuckDuckGoSearchRun()

intent_prompt = PromptTemplate.from_template("""
Classify the user's message into one word:

CHAT
EMAIL
CALENDAR
NEWS

Guidelines:
-EMAIL : If the user is asking to send an email, or asking about email related tasks like writing an email, scheduling an email, etc.
-CALENDAR : If the user is asking to schedule an event, set a reminder, or anything related to calendar.
-NEWS : If the user is asking about current news, headlines, or anything related to news and current events.
-CHAT : For all other messages that do not fit into the above categories, classify as CHAT.


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
- Do not add explanations
- ALWAYS include all three keys
- Keep the body under 120 words
- If any value is missing, leave it blank but keep the key
- End the body with: 

Best regards,
{sender_name}


Return EXACTLY in this format:

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
    "https://www.googleapis.com/auth/gmail.send",
    "https://www.googleapis.com/auth/gmail.readonly"
]

from google.oauth2.credentials import Credentials
from google.auth.transport.requests import Request

def get_services():
    creds = None

    # Check if token.json exists
    if os.path.exists("token.json"):
        creds = Credentials.from_authorized_user_file("token.json", SCOPES)

    if creds and creds.expired and creds.refresh_token:
        creds.refresh(Request())

    # If no valid credentials, do login
    if not creds or not creds.valid:
        flow = InstalledAppFlow.from_client_secrets_file(
            "credentials.json", SCOPES
        )
        creds = flow.run_local_server(port=0)

        # Save credentials
        with open("token.json", "w") as token:
            token.write(creds.to_json())

    calendar_service = build("calendar", "v3", credentials=creds)
    gmail_service = build("gmail", "v1", credentials=creds)

    profile = gmail_service.users().getProfile(userId="me").execute()
    sender_email = profile.get("emailAddress")

    return calendar_service, gmail_service, sender_email

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

def extract_email_from_text(text):
    import re
    email_pattern = r'[\w\.-]+@[\w\.-]+\.\w+'
    matches = re.findall(email_pattern, text)
    return matches[0] if matches else None

def email_tool(user_input):
    global pending_email

    # If user confirms sending
    if pending_email and user_input.strip().upper() == "CONFIRM":
        _, gmail_service, _ = get_services()

        send_email(
            gmail_service,
            pending_email["to"],
            pending_email["subject"],
            pending_email["body"]
        )

        sent_to = pending_email["to"]
        pending_email = None

        return f"Email successfully sent to {sent_to}"

   
    calendar_service, gmail_service, sender_email = get_services()

    sender_name = sender_email.split("@")[0]

# STEP 1: Extract email using regex (NOT LLM)
    to = extract_email_from_text(user_input)

# STEP 2: Generate subject + body using LLM
    details_text = email_extract_chain.invoke({
        "input": user_input,
        "sender_name": sender_name
        })

    data = parse_details(details_text)

    subject = data.get("subject", "No Subject")
    body = data.get("body", "")

    if not to:
        return "Whom should I send the email to? Please provide the recipient email address."

    email_pattern = r'^[\w\.-]+@[\w\.-]+\.\w+$'
    if not re.match(email_pattern, to):
        return "Invalid email address format."

    pending_email = {
        "to": to,
        "subject": subject,
        "body": body
    }

    return f"""
Draft Email:

To: {to}
Subject: {subject}

{body}

Type CONFIRM to send this email.
"""
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

    query = user_input
    results = search.run(query)

    if not results:
        return f"Sorry I couldn't fetch news for {location}."

    summary_prompt = f"""
    Summarize the following news results in 3-4 sentences: \n{results}
    keep it concise and informative and in bullet  points."""
    summary = model.invoke(summary_prompt)
    

    return summary.content



def normal_chat(user_input):
    today = datetime.date.today()
    system_msg = HumanMessage(
        content=f"Today's date is {today}. Answer the user's question."
    )
    chat_history = [system_msg]
    chat_history.append(HumanMessage(content=user_input))
    response = model.invoke(chat_history)
    chat_history.append(AIMessage(content=response.content))
    return response.content


print("AI Assistant Started (type 'exit' to quit)\n")
calendar_service, gmail_service, sender_email = get_services()

while True:
    user_input = input("You: ")

    if user_input.lower() == "exit":
        break

    intent = intent_chain.invoke({"input": user_input}).strip().upper()

    if intent not in ["CHAT", "EMAIL", "CALENDAR", "NEWS"]:
        intent = "CHAT"

    #Allow CONFIRM to trigger email sending
    if intent == "EMAIL" or (pending_email and user_input.strip().upper() == "CONFIRM"):
        reply = email_tool(user_input)

    elif intent == "CALENDAR":
        reply = calendar_tool(user_input)

    elif intent == "NEWS":
        reply = news_tool(user_input)

    else:
        reply = normal_chat(user_input)

    print("AI:", reply)

    ## build ollama model
    ## think about making agent have access to you email and he can make a reply msg to email and when you open gmail own your own then there should be a gmail send button and when you click that button the email should be sent.