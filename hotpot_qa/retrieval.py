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

# Step 1: Load the JSON file into a pandas DataFrame
file_path = './hotpot_train_v1.1.json'
df = pd.read_json(file_path)

# Step 2: Prepare the Text Data
def prepare_documents(df):
    documents = []
    for _, row in tqdm(df.iterrows(), total=df.shape[0], desc="Preparing Documents"):
        context = row['context']
        if isinstance(context, list):
            context_text = " ".join([" ".join(doc[1]) for doc in context if isinstance(doc, list) and len(doc) > 1])
            documents.append(Document(page_content=context_text, metadata={"id": row['_id']}))
    return documents

documents = prepare_documents(df)

with open("documents.pickle", 'wb') as fw:
    pickle.dump(documents, fw)

with open("documents.pickle", 'rb') as fw:
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

with open("split_documents.pickle", 'wb') as fw:
    pickle.dump(split_documents, fw)

with open("split_documents.pickle", 'rb') as fw:
    split_documents = pickle.load(fw)

print(len(documents))
print(len(split_documents))

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
vectorstore.save_local("./faiss")
print("FAISS index saved to disk")

vectorstore = FAISS.load_local('./faiss', embeddings_model, allow_dangerous_deserialization=True)
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
    print("First relevant document:", docs[0].page_content)
    print("First document metadata:", docs[0].metadata)
