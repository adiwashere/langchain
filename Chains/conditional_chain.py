from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.runnables import RunnableParallel , RunnableBranch, RunnableLambda
from dotenv import load_dotenv
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.output_parsers import PydanticOutputParser
from pydantic import BaseModel, Field
from typing import Literal

load_dotenv()

model = ChatGoogleGenerativeAI(model="gemini-2.5-flash", temperature=0.2)  

parser = StrOutputParser()

class feedback(BaseModel):
    sentiment: Literal['positive', 'negative'] = Field(description="The sentiment of the feedback text")

parser2 = PydanticOutputParser(pydantic_object=feedback)

prompt1 = PromptTemplate(
    template = 'classify the sentiment of following feedback text as positive or negative \n {text} \n {format_instruction}.',
    input_variables=['text'],
    partial_variables={'format_instruction': parser2.get_format_instructions()}
)

classifier_chain = prompt1 | model | parser2

prompt2 = PromptTemplate(
    template='Reply directly to the customer with ONE short positive message. Do not explain. Feedback: {text}',
    input_variables=['text']
)

prompt3 = PromptTemplate(
    template='Reply directly to the customer with ONE short polite apology and solution. Do not explain. Feedback: {text}',
    input_variables=['text']
)

#if sentiment is pos then run this chain else run that chain
branch_chain = RunnableBranch(
    (lambda x: x.sentiment == 'positive', prompt2 | model | parser),
    (lambda x: x.sentiment == 'negative', prompt3 | model | parser),
    RunnableLambda(lambda x:"could not find sentiment" )#default case)
)


chain = classifier_chain | branch_chain

result = chain.invoke({'text':'The product quality is outstandin!'})
print(result)