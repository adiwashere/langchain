# from langchain_core.runnables import RunnableLambda

# def word_count(text):
#     return len(text.split())

# runn_word_counter = RunnableLambda(word_count)

# result = runn_word_counter.invoke("hello world")
# print(result)

from langchain_google_genai import ChatGoogleGenerativeAI
from dotenv import load_dotenv
load_dotenv()
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnableLambda, RunnableParallel, RunnablePassthrough
from langchain_huggingface import ChatHuggingFace, HuggingFaceEndpoint

llm = HuggingFaceEndpoint(
    repo_id="google/gemma-2-9b-it",
    task="text-generation",
    temperature=0.1
)

model = ChatHuggingFace(llm=llm)

# model = ChatGoogleGenerativeAI(model="gemini-2.5-flash", temperature=0.2)
def word_count(text):
    return len(text.split())

prompt = PromptTemplate(
    template='Write only one joke about a {topic}.',
    input_variables=['topic']
)

parser = StrOutputParser()

joke_gen_chain= prompt | model | parser

parallel_chain = RunnableParallel({
    'joke':   RunnablePassthrough(),
    'word_count': RunnableLambda(word_count)
})

final_chain= joke_gen_chain | parallel_chain

result = final_chain.invoke({'topic':'space exploration'})
print(result)