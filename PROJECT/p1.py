from langchain_google_genai import ChatGoogleGenerativeAI
from dotenv import load_dotenv
from langchain_core.messages import HumanMessage,AIMessage
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import (
    RunnablePassthrough,
    RunnableParallel,
    RunnableLambda,
)

from langchain_huggingface import ChatHuggingFace, HuggingFaceEndpoint

load_dotenv()

# Initialize Gemini model
# model = ChatGoogleGenerativeAI(
#     model="gemini-2.5-pro",
#     temperature=0.2
# )

llm = HuggingFaceEndpoint(
    repo_id="google/gemma-2-9b-it",
    task="text-generation",
    temperature=0.1
)

model = ChatHuggingFace(llm=llm)


chat_history = []

while True:
    user_input = input("You: ")

    if user_input == "exit":
        break
    chat_history.append(HumanMessage(content=user_input))

    result = model.invoke(chat_history[-4:])

    chat_history.append(AIMessage(content=result.content))
    print("AI:", result.content)

    # print only text
    print("Chat history:")
    