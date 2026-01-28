from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.runnables import RunnableParallel
from dotenv import load_dotenv
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser

load_dotenv()

model1 = ChatGoogleGenerativeAI(model="gemini-2.5-flash", temperature=0.2)   
model2 =  ChatGoogleGenerativeAI(model="gemini-2.5-flash", temperature=0.2)   

prompt1 = PromptTemplate(
    template='generate short and simple notes \n {text}.',
    input_variables=['text']
)

prompt2 = PromptTemplate(
    template='Generate 5 short answer question from the following text\n{text}',  
    input_variables=['text']
)

prompt3 = PromptTemplate(
    template = ' Merge the provided notes and questions into a single comprehensive study guide.\n Notes: {notes}\n Questions: {questions}',
    input_variables=['notes', 'questions']
)
parser = StrOutputParser()

parallel_chain = RunnableParallel({
    'notes':   prompt1 | model1 | parser,
    'questions': prompt2 | model2 | parser
}
)

merge_chain = prompt3 | model1 | parser

chain = parallel_chain | merge_chain

result = chain.invoke({'text':'Quantum computing is a type of computation that harnesses the collective properties of quantum states, such as superposition, interference, and entanglement, to perform calculations. Unlike classical computers that use bits as the smallest unit of data (0s and 1s), quantum computers use quantum bits or qubits. Qubits can exist in multiple states simultaneously, allowing quantum computers to process a vast number of possibilities at once. This capability makes them particularly well-suited for certain complex problems, such as factoring large numbers, simulating molecular structures, and optimizing complex systems. However, building practical and scalable quantum computers remains a significant scientific and engineering challenge due to issues like qubit coherence and error rates.'})
print(result)

