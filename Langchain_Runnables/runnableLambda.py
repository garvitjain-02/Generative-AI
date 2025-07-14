from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import PromptTemplate
from langchain.schema.runnable import RunnableSequence,RunnableParallel,RunnablePassthrough,RunnableLambda
from dotenv import load_dotenv

load_dotenv()

prompt = PromptTemplate(
    template='Write a joke about {topic}',
    input_variables=['topic']
)
def wordcount(text):
    return len(text.split())

model = ChatGoogleGenerativeAI(model="gemini-2.0-flash")

parser = StrOutputParser()

joke_generator = RunnableSequence(prompt | model | parser)
parallelchain = RunnableParallel(
    {
        'joke':RunnablePassthrough(),
        'words':RunnableLambda(lambda x: len(x.split()))
        # 'words':RunnableLambda(wordcount)
    }
)
finalchain = RunnableSequence(joke_generator | parallelchain)
result = finalchain.invoke({'topic':'shoes'})

finalresult="""Joke: {} \n Word Count: {}""".format(result['joke'],result['words'])
print(finalresult)