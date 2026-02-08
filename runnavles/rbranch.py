from dotenv import load_dotenv
load_dotenv()

from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnableLambda, RunnablePassthrough, RunnableBranch
from langchain_huggingface import ChatHuggingFace, HuggingFaceEndpoint

llm = HuggingFaceEndpoint(
    repo_id="google/gemma-2-9b-it",
    task="text-generation",
    temperature=0
)

model = ChatHuggingFace(llm=llm)

prompt1 = PromptTemplate(
    template='Write a detailed report on {topic}.',
    input_variables=['topic']
)

prompt2 = PromptTemplate(
    template='Summarize the following text:\n{text}',
    input_variables=['text']
)

parser = StrOutputParser()

report_gen_chain = prompt1 | model | parser

# function instead of lambda
def is_long_text(text):
    return len(text) > 500

# convert string → dict for prompt2
def to_text_dict(text):
    return {"text": text}

branch_chain = RunnableBranch(
    (is_long_text, to_text_dict | prompt2 | model | parser),
    RunnablePassthrough()
)

final_chain = report_gen_chain | branch_chain

print(final_chain.invoke({'topic': 'Russia vs Ukraine'}))