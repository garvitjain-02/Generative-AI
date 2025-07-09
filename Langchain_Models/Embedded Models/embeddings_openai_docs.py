from langchain_openai import OpenAIEmbeddings
from dotenv import load_dotenv

load_dotenv()

# Creates embedding for many sentences
embedding = OpenAIEmbeddings(model='text-embedding-3-large', dimensions=32)
documents=[
    "delhi is good",
    "alwar is best"
]
result = embedding.embed_documents(documents)
print(str(result))