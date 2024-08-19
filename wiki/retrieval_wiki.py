import pickle
import pandas as pd
from tqdm import tqdm
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_community.vectorstores.faiss import DistanceStrategy
from langchain.docstore.document import Document

import warnings
warnings.filterwarnings('ignore')

import os
os.environ["CUDA_VISIBLE_DEVICES"]= "7"

# Step 1: Load the Parquet file into a pandas DataFrame
file_path = './wiki_abstract_with_vector.parquet'
df = pd.read_parquet(file_path)

# Step 2: Prepare the Text Data
def prepare_documents(df):
    documents = []
    for _, row in tqdm(df.iterrows(), total=df.shape[0], desc="Preparing Documents"):
        body_decoded = row['body'].decode('utf-8', errors='ignore') if isinstance(row['body'], bytes) else row['body']
        documents.append(Document(page_content=body_decoded, metadata={"id": row['id']}))
    return documents

documents = prepare_documents(df)

# Optional: Save and load documents
with open("documents_wiki.pickle", 'wb') as fw:
    pickle.dump(documents, fw)

with open("documents_wiki.pickle", 'rb') as fw:
    documents = pickle.load(fw)

# Step 3: Split the Documents into Chunks
text_splitter = RecursiveCharacterTextSplitter.from_tiktoken_encoder(
    chunk_size=1000,
    chunk_overlap=200,
    encoding_name='cl100k_base'
)

split_documents = []
for doc in tqdm(documents, desc="Splitting Documents", total=len(documents)):
    split_documents.extend(text_splitter.split_documents([doc]))

# Optional: Save and load split documents
with open("split_documents_wiki.pickle", 'wb') as fw:
    pickle.dump(split_documents, fw)

with open("split_documents_wiki.pickle", 'rb') as fw:
    split_documents = pickle.load(fw)

print(f"Number of documents: {len(documents)}")
print(f"Number of split documents: {len(split_documents)}")

# Step 4: Generate Embeddings
model_name = "intfloat/multilingual-e5-large-instruct"
embeddings_model = HuggingFaceEmbeddings(
    model_name=model_name,
    model_kwargs={"device": "cuda"}, 
    encode_kwargs={"normalize_embeddings": True},
)
print("Embedding models imported")

# Step 5: Create FAISS Vector Store
vectorstore = FAISS.from_documents(
    documents=split_documents,
    embedding=embeddings_model,
    distance_strategy=DistanceStrategy.COSINE
)
vectorstore.save_local("./faiss_wiki")
print("FAISS index saved to disk")

# Load the FAISS index
vectorstore = FAISS.load_local('./faiss_wiki', embeddings_model, allow_dangerous_deserialization=True)
print("FAISS index loaded from disk")

# Example Query and Retrieval
query = "What is the history of Arthur's Magazine?"
retriever = vectorstore.as_retriever(
    search_type='mmr',
    search_kwargs={'k': 5, 'fetch_k': 50}
)

docs = retriever.get_relevant_documents(query)
print(f"Number of relevant documents: {len(docs)}")
if docs:
    print(docs)
