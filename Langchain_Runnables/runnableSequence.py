from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import PromptTemplate
from langchain.schema.runnable import RunnableSequence
from dotenv import load_dotenv


load_dotenv()

prompt = PromptTemplate(
    template='Write a joke about {topic}',
    input_variables=['topic']
)
prompt2 = PromptTemplate(
    template='explain this joke-> {joke}',
    input_variables=['joke']
)

model = ChatGoogleGenerativeAI(model="gemini-2.0-flash")

parser = StrOutputParser()

chain = RunnableSequence(prompt , model , parser , prompt2 , model , parser)
result = chain.invoke({'topic':'girl'})
print(result)