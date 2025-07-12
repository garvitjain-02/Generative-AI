from langchain_huggingface import ChatHuggingFace, HuggingFaceEndpoint
from dotenv import load_dotenv
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser

load_dotenv()

llm1 = HuggingFaceEndpoint(
    repo_id="google/gemma-2-2b-it",
    task="text-generation"
)
model = ChatHuggingFace (llm=llm1)


#prompt-1 Detailed report
template1=PromptTemplate(
    template='Write a detailed report on {topic}',
    input_variables=['topic']
)

#prompt-2 summary
template2=PromptTemplate(
    template='Write a 5 line summary on following text. \n {text}',
    input_variables=['text']
)

parser = StrOutputParser()

chain = template1 | model | parser | template2 | model | parser
result = chain.invoke({'topic':'Machine Learning'})
print(result)


# below is code for normal output w/o output parser

# prompt1=template1.invoke({'topic':'Blackhole'})
# result1 = model.invoke(prompt1)

# prompt2=template2.invoke({'text':result1.content})
# result2 = model.invoke(prompt2)

# print(result2.content)