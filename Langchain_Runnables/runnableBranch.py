from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import PromptTemplate
from langchain.schema.runnable import RunnableSequence,RunnableParallel,RunnablePassthrough,RunnableLambda,RunnableBranch
from dotenv import load_dotenv

load_dotenv()

prompt1 = PromptTemplate(
    template='Write a report about topic: {topic}',
    input_variables=['topic']
)
prompt2 = PromptTemplate(
    template='summarize the following text: {text}',
    input_variables=['text']
)

model = ChatGoogleGenerativeAI(model="gemini-2.0-flash")

parser = StrOutputParser()

reportchain=RunnableSequence(prompt1 | model | parser)
branchchain = RunnableBranch(
    (lambda x:len(x.split())>500,RunnableSequence(prompt2 | model | parser)),
    RunnablePassthrough()
)

finalchain = reportchain | branchchain
result = finalchain.invoke({'topic':'Machine Learning'})
print(result)