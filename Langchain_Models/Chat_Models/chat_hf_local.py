from langchain_huggingface import ChatHuggingFace, HuggingFacePipeline

llm1 = HuggingFacePipeline.from_model_id(
    model_id="google/gemma-2-2b-it",
    task="text-generation",
    pipeline_kwargs=dict (
        temperature=0.5,
        max_new_tokens=100
    )
)

model = ChatHuggingFace(llm=llm1)
result = model.invoke("top 10 places to visit in india in monsoon")
print(result.content)
