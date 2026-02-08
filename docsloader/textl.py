from langchain_community.document_loaders import TextLoader
from langchain_huggingface import ChatHuggingFace, HuggingFaceEndpoint
from dotenv import load_dotenv
from langchain_core.messages import HumanMessage, AIMessage
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
load_dotenv()

llm = HuggingFaceEndpoint(
    repo_id="google/gemma-2-9b-it",
    task="text-generation",
    temperature=0
)

model = ChatHuggingFace(llm=llm)

parser = StrOutputParser()

promp = PromptTemplate(
    template='Write a summary on following poem{topic}.',
    input_variables=['topic']
)

loader = TextLoader("cricket.txt", encoding="utf-8")
documents = loader.load()

print(documents)
print(type(documents)  ) # makes a list of documents 
print(len(documents)) # you will get only one document
print(documents[0]) # list ka pahela item

chain = promp | model | parser

result = chain.invoke({'topic':documents[0].page_content})
print("\n ----------------\n",result)