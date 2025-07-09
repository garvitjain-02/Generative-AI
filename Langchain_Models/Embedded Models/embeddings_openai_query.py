from langchain_openai import OpenAIEmbeddings
from dotenv import load_dotenv

load_dotenv()

# Creates embedding for single sentence
embedding = OpenAIEmbeddings(model='text-embedding-3-large', dimensions=32)
result = embedding.embed_query("Delhi is large city")
print(str(result))