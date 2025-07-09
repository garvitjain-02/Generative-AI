from langchain_huggingface import HuggingFaceEmbeddings

embedding = HuggingFaceEmbeddings(model_name= 'sentence-transformers/all-MiniLM-L6-v2')

document = [
    "I alive in Alwar",
    "i am garvit",
    "hello good morning"
]
vector = embedding.embed_documents(document)

print(str(vector))