from langchain_core.prompts import PromptTemplate

intent_prompt = PromptTemplate.from_template("""
Classify the user's message into one word:

CHAT
EMAIL
CALENDAR
NEWS

Message: {input}

Answer:
""")

calendar_extract_prompt = PromptTemplate.from_template("""
Extract event details from this message.

Message: {input}

Return EXACTLY in this format:

title:
date:
time:
duration_minutes:
""")

email_extract_prompt = PromptTemplate.from_template("""
You are an assistant that writes professional emails.

Return EXACTLY in this format:

to:
subject:
body:

Message: {input}
""")

location_extract_prompt = PromptTemplate.from_template("""
Extract location from this message.

If none mentioned return India.

Message: {input}
""")