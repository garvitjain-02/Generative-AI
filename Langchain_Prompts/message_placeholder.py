from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder

#chat template
chat_temp=ChatPromptTemplate([
    ('system','You are a good assistant'),
    MessagesPlaceholder(variable_name='chat_history'),
    ('human','{query}')
])

chat_history=[]

#load chat history
with open('chat_history.txt') as f:
    chat_history.extend(f.readlines())

print(chat_history)

prompt = chat_temp.invoke({'chat_history':chat_history,'query':"Where is my refund?"})

print(prompt)