from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.messages import HumanMessage, AIMessage

#chat template
chat_template = ChatPromptTemplate([
    ('system', 'You are a helpful assistant. Answer politely.'),
    MessagesPlaceholder(variable_name='chat_history'),
    ('human', '{query}')
]
)
#chat history variable
chat_history = []

#load chat history
with open('chat_history.txt') as file:
    chat_history.extend(file.readlines())


print(chat_history)


final_prompt= chat_template.invoke({
    'chat_history': chat_history,
    'query': 'Were is my package?'
})  

print(final_prompt)
