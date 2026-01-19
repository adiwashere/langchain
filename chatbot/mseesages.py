from langchain_core.messages import HumanMessage, AIMessage
from langchain_huggingface import ChatHuggingFace, HuggingFaceEndpoint
from dotenv import load_dotenv  

load_dotenv()

llm = HuggingFaceEndpoint(
    repo_id="google/gemma-2-9b-it", 
    task="text-generation",
    temperature=0.1
)

model = ChatHuggingFace(llm=llm)

messages = [
    HumanMessage(
        content="You are a helpful assistant. Answer politely.\n\nHello, how are you?"
    )
]

result = model.invoke(messages)
messages.append(AIMessage(content=result.content))

# print only text
for msg in messages:
    print(msg.content)
