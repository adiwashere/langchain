from langchain_google_genai import ChatGoogleGenerativeAI
from dotenv import load_dotenv
load_dotenv()
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser


model = ChatGoogleGenerativeAI(model="gemini-2.5-flash", temperature=0.2)   

prompt = PromptTemplate(
    template='Write only one joke about a {topic}.',
    input_variables=['topic']
)
prompt2 = PromptTemplate(
    template= 'Explain thr following joke {text}',
    input_variables=['text']
)

parser = StrOutputParser()
chain = prompt | model | parser | prompt2 | model | parser
result = chain.invoke({'topic':'space exploration'} )
print("+++++++\n",result)