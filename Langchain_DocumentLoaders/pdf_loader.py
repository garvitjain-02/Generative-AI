from langchain_community.document_loaders import PyPDFLoader

loader=PyPDFLoader('1409.3215v3.pdf')
docs=loader.load()
print(len(docs))
print(docs[0].page_content)
print(docs[0].metadata)
