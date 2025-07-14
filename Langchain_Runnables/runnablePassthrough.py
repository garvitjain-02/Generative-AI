from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import PromptTemplate
from langchain.schema.runnable import RunnableSequence,RunnableParallel,RunnablePassthrough
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

joke_generator = RunnableSequence(prompt | model | parser)
parallelchain = RunnableParallel(
    {
        'jokes':RunnablePassthrough(),
        'explanation':RunnableSequence(prompt2 | model | parser)
    }
)

finalchain=RunnableSequence(joke_generator | parallelchain)
result = finalchain.invoke({'topic':'cricket'})
print(result)