# rag.py

import os
from langchain_community.llms import Ollama
from langchain_community.document_loaders import TextLoader, PyPDFLoader, UnstructuredWordDocumentLoader
from langchain_community.embeddings import SentenceTransformerEmbeddings
from langchain_community.vectorstores import Chroma
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.chains import RetrievalQA
from langchain.docstore.document import Document

# Load documents from /data
def load_documents():
    all_docs = []
    for file in os.listdir("data"):
        path = os.path.join("data", file)
        try:
            if file.endswith(".txt"):
                loader = TextLoader(path)
            elif file.endswith(".pdf"):
                loader = PyPDFLoader(path)
            elif file.endswith(".docx"):
                loader = UnstructuredWordDocumentLoader(path)
            else:
                print(f"⚠️ Unsupported file: {file}")
                continue
            docs = loader.load()
            all_docs.extend(docs)
        except Exception as e:
            print(f"❌ Failed to load {file}: {e}")
    return all_docs

# Step 1: Load and chunk documents
documents = load_documents()
if not documents:
    print("❌ No documents found in /data.")
    exit()

text_splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)
chunked_docs = text_splitter.split_documents(documents)

# Step 2: Embed & store
embedding = SentenceTransformerEmbeddings(model_name="all-MiniLM-L6-v2")
vectorstore = Chroma.from_documents(chunked_docs, embedding=embedding)

# Step 3: Setup LLM and QA chain
llm = Ollama(model="tinyllama")
qa = RetrievalQA.from_chain_type(
    llm=llm,
    retriever=vectorstore.as_retriever(),
    return_source_documents=True
)

# Step 4: Ask questions
while True:
    query = input("\nAsk your question (or type 'exit'): ")
    if query.lower() == "exit":
        break
    result = qa(query)
    print("\n Answer:", result['result'])
    print("\n Source Chunks:\n")
    for doc in result["source_documents"]:
        print("➤", doc.page_content[:300], "...\n")
