from langchain_google_genai import GoogleGenerativeAIEmbeddings
from dotenv import load_dotenv
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np

load_dotenv()

embeddings = GoogleGenerativeAIEmbeddings(model='models/text-embedding-004')

document = [
    "Virat kohli is indian cricketer known for aggressive batting",
    "Ms Dhoni, former indian captain famous for calm nature and finishing skills",
    "Sachin tendulkar is known for short height",
    "The sky is blue and clouds are aggressive",
    "Bumrah is indian fast bowler with unorthodox action"
]
query = "who is aggressive cricketer?"
doc_vector = embeddings.embed_documents(document)
query_vector = embeddings.embed_query(query)

scores = cosine_similarity([query_vector],doc_vector)

index= np.argmax(np.array(scores))
print(document[index])
print('Similarity scores is: ', scores[index])