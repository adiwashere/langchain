from langchain_google_genai import ChatGoogleGenerativeAI
from dotenv import load_dotenv
load_dotenv()
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough, RunnableParallel

model = ChatGoogleGenerativeAI(model="gemini-2.5-flash", temperature=0.2)

parser = StrOutputParser()

prompt = PromptTemplate(
    template='Write only one joke about a {topic}.',
    input_variables=['topic']
)
prompt2 = PromptTemplate(
    template= 'Explain thr following joke {text}',
    input_variables=['text']
)

joke_gen_chain= prompt | model | parser

parallel_chain = RunnableParallel({
    'joke':   RunnablePassthrough(),
    'explain': prompt2 | model | parser
})

final_chain= joke_gen_chain | parallel_chain

result = final_chain.invoke({'topic':'space exploration'})
print("+++++++\n",result)