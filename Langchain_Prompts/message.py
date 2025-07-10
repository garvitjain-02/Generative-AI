from langchain_google_genai import GoogleGenerativeAI
from langchain_core.messages import SystemMessage,AIMessage,HumanMessage
from dotenv import load_dotenv

load_dotenv()

model = GoogleGenerativeAI(
    model="gemini-2.0-flash"
)

messages=[
    SystemMessage(content="You are helpful assistant and have knowledge of Software development"),
    HumanMessage(content="what is best programming language to learn in 2025?")
]
result= model.invoke(messages)

messages.append( AIMessage(content=result) )

print(messages)