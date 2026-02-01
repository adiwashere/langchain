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
    




# # ------------------ PROMPTS ------------------

# chat_prompt = PromptTemplate(
#     input_variables=["question"],
#     template="Answer the user's question clearly:\n{question}"
# )

# intent_prompt = PromptTemplate(
#     input_variables=["question"],
#     template="""
# Classify the user's intent.
# Return ONLY one word:
# - chat
# - task

# Question: {question}
# """
# )

# # ------------------ CHAINS ------------------

# intent_chain = intent_prompt | model | StrOutputParser()
# chat_chain = chat_prompt | model | StrOutputParser()

# # Dummy task executor (Day 1 level)
# def execute_task(question: str) -> str:
#     return f"[TASK EXECUTOR] Task received: {question}"

# task_chain = RunnableLambda(
#     lambda x: execute_task(x["question"])
# )

# # ------------------ ROUTING LOGIC ------------------

# def route(input_dict):
#     if input_dict["intent"].strip().lower() == "task":
#         return task_chain.invoke(input_dict)
#     return chat_chain.invoke(input_dict)

# router = RunnableLambda(route)

# # ------------------ PIPELINE ------------------

# chain = (
#     RunnableParallel(
#         question=RunnablePassthrough(),
#         intent=intent_chain
#     )
#     | router
# )

# # ------------------ RUN ------------------

# if __name__ == "__main__":
#     while True:
#         user_input = input("You: ")
#         if user_input.lower() in ["exit", "quit"]:
#             break
#         response = chain.invoke(user_input)
#         print("Bot:", response)
