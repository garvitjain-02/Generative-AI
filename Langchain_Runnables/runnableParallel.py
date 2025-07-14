from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import PromptTemplate
from langchain.schema.runnable import RunnableSequence,RunnableParallel
from dotenv import load_dotenv


load_dotenv()

prompt1 = PromptTemplate(
    template='Generate a tweet about {topic}',
    input_variables=['topic']
)
prompt2 = PromptTemplate(
    template='Generate a linkedin post about {topic}',
    input_variables=['topic']
)

parser=StrOutputParser()
model = ChatGoogleGenerativeAI(model="gemini-2.0-flash")

chain=RunnableParallel({
    'tweet':RunnableSequence(prompt1|model|parser),
    'linkedin':RunnableSequence(prompt2|model|parser)
})

result = chain.invoke({'topic':'Seq2Seq Architecture'})
print(result['tweet'])
print('\n')
print(result['linkedin'])