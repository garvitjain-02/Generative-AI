from langchain_huggingface import ChatHuggingFace, HuggingFaceEndpoint
from dotenv import load_dotenv
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import PydanticOutputParser
from pydantic import BaseModel, Field

load_dotenv()

llm1 = HuggingFaceEndpoint(
    repo_id="google/gemma-2-2b-it",
    task="text-generation"
)
model = ChatHuggingFace (llm=llm1)

class Person(BaseModel):
    name: str = Field(description="name of the person")
    age: int = Field(gt=18,description="age of the person")
    city: str = Field(description='Name of the city person belongs to')


parser= PydanticOutputParser(pydantic_object=Person)

template = PromptTemplate(
    template='Generate the name, age , city of fictional {place} person \n {format_instruction}',
    input_variables=['place'],
    partial_variables={'format_instruction': parser.get_format_instructions()}
)

# prompt = template.invoke({'place':'Indian'})
# result = model.invoke(prompt)
# final_result = parser.parse(result.content)


chain = template | model | parser
final_result = chain.invoke({'place':'Indian'})
print(final_result)