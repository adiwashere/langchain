from langchain_google_genai import ChatGoogleGenerativeAI

from dotenv import load_dotenv
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser

load_dotenv()

model = ChatGoogleGenerativeAI(model="gemini-2.5-flash", temperature=0.2)   

prompt1 = PromptTemplate(
    template='Generate a detail report on {topic}.',
    input_variables=['topic']
)

prompt2 = PromptTemplate(
    template='Summarize the following report in 5 small points:\n{report}',
    input_variables=['report']
)
parser = StrOutputParser()

chain = prompt1 | model | prompt2 | model | parser
result = chain.invoke({'topic':'space exploration'})
print(result)