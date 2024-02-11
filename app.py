Create .env file with the following vars



```

# Install required packages
import sys

#!conda install --yes --prefix {sys.prefix} numpy
!conda install --yes --prefix {sys.prefix} -c conda-forge langchain
!conda install --yes --prefix {sys.prefix} python-dotenv
!conda install --yes --prefix {sys.prefix} -c conda-forge openai 
!conda install --yes --prefix {sys.prefix} -c conda-forge tiktoken
!conda install --yes --prefix {sys.prefix} -c conda-forge pypdf
!conda install --yes --prefix {sys.prefix} -c conda-forge chromadb
import os
from dotenv import load_dotenv
from langchain.document_loaders import PyPDFLoader # for loading the pdf
from langchain.embeddings import OpenAIEmbeddings
from langchain.vectorstores import Chroma

load_dotenv()

loader = PyPDFLoader('power-platform-2023-release-wave-1-plan.pdf')

# combine all text
pages = loader.load_and_split()

embeddings = OpenAIEmbeddings(deployment="text-embedding-ada-002", chunk_size=1)

vectordb = Chroma.from_documents(pages, embedding=embeddings,
                                 persist_directory="./vectordb")

#vectordb.persist()



from langchain.chains import ConversationalRetrievalChain
from langchain.chat_models import AzureChatOpenAI


llm = AzureChatOpenAI(deployment_name="gpt-35-turbo", temperature=0.9)

pdf_qa = ConversationalRetrievalChain.from_llm(llm, vectordb.as_retriever())

query = "When can we expect Microsoft Dataverse to go in Public preview?"

result = pdf_qa({"question": query, "chat_history": ""})

print("Answer:")
print(result["answer"])

query = "List all of the features going into public preview in May 2023?"

result = pdf_qa({"question": query, "chat_history": ""})

print("Answer:")
print(result["answer"])
query = "List all of the features going into public preview in Sep 2023?"

result = pdf_qa({"question": query, "chat_history": ""})

print("Answer:")
print(result["answer"])
