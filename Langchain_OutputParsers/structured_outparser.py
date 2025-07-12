from langchain_huggingface import ChatHuggingFace, HuggingFaceEndpoint
from dotenv import load_dotenv
from langchain_core.prompts import PromptTemplate
from langchain.output_parsers import StructuredOutputParser, ResponseSchema

load_dotenv()

llm1 = HuggingFaceEndpoint(
    repo_id="google/gemma-2-2b-it",
    task="text-generation"
)
model = ChatHuggingFace (llm=llm1)

schema=[
    ResponseSchema(name='fact_1', description='Fact 1 about the given topic'),
    ResponseSchema(name='fact_2', description='Fact 2 about the given topic'),
    ResponseSchema(name='fact_3', description='Fact 2 about the given topic')
]

parser = StructuredOutputParser.from_response_schemas(schema)

template= PromptTemplate(
    template='Give 3 facts about the {topic} \n {format_instruction}',
    input_variables=['topic'],
    partial_variables = {'format_instruction':parser.get_format_instructions()}

)

# prompt = template.invoke({'topic':'Machine learning'})
# result = model.invoke(prompt)
# final_result = parser.parse(result.content)

chain = template | model | parser
final_result = chain.invoke({'topic':'Trees'})
print(final_result)