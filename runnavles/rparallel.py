from langchain_google_genai import ChatGoogleGenerativeAI
from dotenv import load_dotenv
load_dotenv()
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnableParallel

model = ChatGoogleGenerativeAI(model="gemini-2.5-flash", temperature=0.2)   

prompt1 = PromptTemplate(
    template='Generate a short report on {topic}.',
    input_variables=['topic']
)

prompt2 = PromptTemplate(
    template='Give 3 short key points about {topic}.',
    input_variables=['topic']
)

parser = StrOutputParser()

parallel_chain = RunnableParallel({
    'report':   prompt1 | model | parser,
    'summary': prompt2 | model | parser
})

result = parallel_chain.invoke({'topic':'space exploration'})
print(result['report'])
print(result['summary'])