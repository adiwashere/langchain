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

load_dotenv()


llm_endpoint = HuggingFaceEndpoint(
    repo_id="google/gemma-2-9b-it",
    task="text-generation",
    temperature=0.2,
)

model = ChatHuggingFace(llm=llm_endpoint)

chat_history = []


intent_prompt = PromptTemplate.from_template("""
Classify the user's message into one word:

CHAT
EMAIL
CALENDAR

Message: {input}

Answer:
""")

intent_chain = intent_prompt | model | StrOutputParser()
#pushed 


email_extract_prompt = PromptTemplate.from_template("""
Extract email details.

Message: {input}

Return format:
to:
subject:
body:
""")

calendar_extract_prompt = PromptTemplate.from_template("""
Extract event details from this message.

Message: {input}

Return EXACTLY in this format:

title:
date:
time:
duration_minutes:

Do not add anything in brackets or extra text.
""")

email_extract_chain = email_extract_prompt | model | StrOutputParser()
calendar_extract_chain = calendar_extract_prompt | model | StrOutputParser()
