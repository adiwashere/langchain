# from langchain_google_genai import GoogleGenerativeAI
# from dotenv import load_dotenv


# load_dotenv()

# model = GoogleGenerativeAI(model_name="gemini-pro", temperature=0)

from langchain_huggingface import ChatHuggingFace, HuggingFaceEndpoint
from dotenv import load_dotenv
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
load_dotenv()

llm= HuggingFaceEndpoint(
    repo_id="google/gemma-2-9b-it",
    task="text-generation",
    temperature=0
)

model = ChatHuggingFace(llm=llm)

# 1st prompt -> detail report
template1= PromptTemplate(
    input_variables=['topic'],
    template='write a detailed report on the topic: {topic}'
)   

#2nd prompt -> summary of report
template2= PromptTemplate(
    template='write a 5 line summary of the following report: {report}',
    input_variables=['report']
)

parser = StrOutputParser() 
chain = template1 | model | parser | template2 | model | parser 

result =chain.invoke({'topic':'climate change'})
print(result)