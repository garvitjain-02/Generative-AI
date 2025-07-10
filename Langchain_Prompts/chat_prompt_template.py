from langchain_core.prompts import ChatPromptTemplate
from langchain_core.messages import SystemMessage,AIMessage,HumanMessage

chat_temp=ChatPromptTemplate([
    ('system','You are a {domain} expert'),
    ('human','Explain me about {topic}')
])

prompt=chat_temp.invoke({'domain':'Cricket','topic':'Googly'})

print(prompt)