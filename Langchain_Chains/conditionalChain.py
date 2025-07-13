from langchain_google_genai import ChatGoogleGenerativeAI
from dotenv import load_dotenv
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser, PydanticOutputParser
from pydantic import BaseModel, Field
from typing import Literal
from langchain.schema.runnable import RunnableBranch,RunnableLambda

load_dotenv()

model = ChatGoogleGenerativeAI(model="gemini-2.0-flash")

class feedback(BaseModel):
    sentiment: Literal['positive','negative'] = Field(description='give the sentiment of the given feedback')

parser = StrOutputParser()
parser2 = PydanticOutputParser(pydantic_object=feedback)
prompt1 = PromptTemplate(
    template='classify the sentiment of the following text into positive,negative \n {feedback} \n {format_instructions}',
    input_variables=['feedback'],
    partial_variables={'format_instructions':parser2.get_format_instructions()}
)

classifier_chain = prompt1 | model | parser2

prompt2= PromptTemplate(
    template = 'Write one appropriate response to this positive feedback-> \n {feedback}',
    input_variables=['feedback']
)
prompt3= PromptTemplate(
    template = 'Write one appropriate response to this negative feedback-> \n {feedback}',
    input_variables=['feedback']
)

branch_chain=RunnableBranch(
    (lambda x:x.sentiment=='positive',prompt2 | model | parser),
    (lambda x:x.sentiment=='negative',prompt3 | model | parser),
    RunnableLambda(lambda x:"Could not find sentiment")
)

final_chain = classifier_chain | branch_chain
result = final_chain.invoke({'feedback':'This is a phone with low battery and not so good screen'})
print(result)