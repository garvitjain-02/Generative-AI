from langchain_google_genai import GoogleGenerativeAI
from dotenv import load_dotenv
import streamlit as st
from langchain_core.prompts import PromptTemplate
from langchain_core.messages import SystemMessage,AIMessage,HumanMessage


load_dotenv()

chat_history=[
    SystemMessage(content="You are helpful assistant."),
]

model = GoogleGenerativeAI(
    model="gemini-2.0-flash"
)

while(True):
    user_input = input("You: ")
    if user_input == 'exit':
        break
    chat_history.append(HumanMessage(content=user_input))
    result = model.invoke(chat_history)
    chat_history.append(AIMessage(content=result))

    print("AI: ",result)

print(chat_history)