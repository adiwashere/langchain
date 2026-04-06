# import re
# from ai.utils.parser import parse_details
# from ai.services.gmail_service import send_email, get_gmail_service
# from ai.model import model
# from ai.prompts import email_extract_prompt
# from langchain_core.output_parsers import StrOutputParser

# email_extract_chain = email_extract_prompt | model | StrOutputParser()

# pending_email = None

# def email_tool(user_input,session):

#     global pending_email

#     if pending_email and user_input.strip().upper() == "CONFIRM":

#         gmail_service, _ = get_gmail_service()

#         send_email(
#             gmail_service,
#             pending_email["to"],
#             pending_email["subject"],
#             pending_email["body"]
#         )

#         sent_to = pending_email["to"]
#         pending_email = None

#         return f"Email successfully sent to {sent_to}"

#     gmail_service, sender_email = get_gmail_service()

#     sender_name = sender_email.split("@")[0]

#     details_text = email_extract_chain.invoke({
#         "input": user_input,
#         "sender_name": sender_name
#     })

#     data = parse_details(details_text)

#     to = data.get("to")
#     subject = data.get("subject", "No Subject")
#     body = data.get("body", "")

#     if not to:
#         return "Provide recipient email."

#     email_pattern = r'^[\w\.-]+@[\w\.-]+\.\w+$'

#     if not re.match(email_pattern, to):
#         return "Invalid email address."

#     pending_email = {
#         "to": to,
#         "subject": subject,
#         "body": body
#     }

#     return f"""
# Draft Email:

# To: {to}
# Subject: {subject}

# {body}

# Type CONFIRM to send.
# """

import re
from ai.utils.parser import parse_details
from ai.services.gmail_service import send_email, get_gmail_service
from ai.model import model
from ai.prompts import email_extract_prompt
from langchain_core.output_parsers import StrOutputParser

email_extract_chain = email_extract_prompt | model | StrOutputParser()

def email_tool(user_input, session):

    #  CONFIRM BLOCK
    if session["pending_email"] and user_input.strip().lower() == "confirm":

        gmail_service, _ = get_gmail_service()

        send_email(
            gmail_service,
            session["pending_email"]["to"],
            session["pending_email"]["subject"],
            session["pending_email"]["body"]
        )

        sent_to = session["pending_email"]["to"]
        session["pending_email"] = None

        return f"Email successfully sent to {sent_to}"

    #  DRAFT CREATION
    gmail_service, sender_email = get_gmail_service()

    sender_name = sender_email.split("@")[0]

    details_text = email_extract_chain.invoke({
        "input": user_input,
        "sender_name": sender_name
    })

    data = parse_details(details_text)

    to = data.get("to")
    subject = data.get("subject", "No Subject")
    body = data.get("body", "")

    if not to:
        return "Provide recipient email."

    email_pattern = r'^[\w\.-]+@[\w\.-]+\.\w+$'

    if not re.match(email_pattern, to):
        return "Invalid email address."

    #  STORE IN SESSION (IMPORTANT)
    session["pending_email"] = {
        "to": to,
        "subject": subject,
        "body": body
    }

    return f"""
Draft Email:

To: {to}
Subject: {subject}

{body}

Type CONFIRM to send.
"""