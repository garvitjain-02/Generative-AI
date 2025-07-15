from langchain_community.document_loaders import WebBaseLoader
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import PromptTemplate
from dotenv import load_dotenv

load_dotenv()

url='https://www.amazon.in/Indian-Garage-Co-Checkered-0121-SH69-05_White/dp/B08ZNCWX7V/?_encoding=UTF8&pd_rd_w=AO9EB&content-id=amzn1.sym.3eb27980-84ee-4cfe-bafb-8634b1863e1c%3Aamzn1.symc.36bd837a-d66d-47d1-8457-ffe9a9f3ddab&pf_rd_p=3eb27980-84ee-4cfe-bafb-8634b1863e1c&pf_rd_r=CVK2D83CXWS5BQY9VY4Q&pd_rd_wg=rYNYf&pd_rd_r=c2d36a3c-87d5-4bbb-ac67-4f92b6c550fb&ref_=pd_hp_d_btf_ci_mcx_mr_hp_atf_m&th=1&psc=1'
loader=WebBaseLoader(url)
docs=loader.load()

model = ChatGoogleGenerativeAI(model="gemini-2.0-flash")
parser = StrOutputParser()
prompt = PromptTemplate(
    template='Answer the following question: {question} from following text: {text}',
    input_variables=['question','text']
)

chain = prompt | model | parser
print(chain.invoke({'question':'who is the manufacturer of this product? ','text':docs[0].page_content}))
