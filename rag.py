import pandas as pd
import torch
from tqdm import tqdm
from langchain_community.vectorstores import FAISS
from langchain_community.vectorstores.utils import DistanceStrategy
from langchain_huggingface import HuggingFaceEmbeddings

import os
os.environ["CUDA_VISIBLE_DEVICES"]= "7"

import warnings
warnings.filterwarnings('ignore')

model_name = "intfloat/multilingual-e5-large-instruct"
embeddings_model = HuggingFaceEmbeddings(
    model_name=model_name,
    model_kwargs={"device": "cuda"}, 
    encode_kwargs={"normalize_embeddings": True},
)

#vectorstore = FAISS.load_local('./hotpot_qa/faiss', embeddings_model, allow_dangerous_deserialization=True)
vectorstore = FAISS.load_local('./wiki/faiss_wiki', embeddings_model, allow_dangerous_deserialization=True)

retriever = vectorstore.as_retriever(
    search_type='mmr',
    #search_kwargs={'k': 2, 'fetch_k': 30}
    search_kwargs={'k': 5, 'fetch_k': 50}
)

##### Load datasets #####
splits = {'test': 'college_mathematics/test-00000-of-00001.parquet', 'validation': 'college_mathematics/validation-00000-of-00001.parquet', 'dev': 'college_mathematics/dev-00000-of-00001.parquet'}
Col_Math = pd.read_parquet("hf://datasets/cais/mmlu/" + splits["test"])

splits = {'test': 'business_ethics/test-00000-of-00001.parquet', 'validation': 'business_ethics/validation-00000-of-00001.parquet', 'dev': 'business_ethics/dev-00000-of-00001.parquet'}
Biz_Ethics = pd.read_parquet("hf://datasets/cais/mmlu/" + splits["test"])

splits = {'test': 'professional_law/test-00000-of-00001.parquet', 'validation': 'professional_law/validation-00000-of-00001.parquet', 'dev': 'professional_law/dev-00000-of-00001.parquet'}
Prf_Law = pd.read_parquet("hf://datasets/cais/mmlu/" + splits["test"])

splits = {'test': 'computer_security/test-00000-of-00001.parquet', 'validation': 'computer_security/validation-00000-of-00001.parquet', 'dev': 'computer_security/dev-00000-of-00001.parquet'}
Com_Secu = pd.read_parquet("hf://datasets/cais/mmlu/" + splits["test"])

splits = {'test': 'anatomy/test-00000-of-00001.parquet', 'validation': 'anatomy/validation-00000-of-00001.parquet', 'dev': 'anatomy/dev-00000-of-00001.parquet'}
Anatomy = pd.read_parquet("hf://datasets/cais/mmlu/" + splits["test"])

splits = {'test': 'astronomy/test-00000-of-00001.parquet', 'validation': 'astronomy/validation-00000-of-00001.parquet', 'dev': 'astronomy/dev-00000-of-00001.parquet'}
Astronomy = pd.read_parquet("hf://datasets/cais/mmlu/" + splits["test"])

splits = {'test': 'marketing/test-00000-of-00001.parquet', 'validation': 'marketing/validation-00000-of-00001.parquet', 'dev': 'marketing/dev-00000-of-00001.parquet'}
Marketing = pd.read_parquet("hf://datasets/cais/mmlu/" + splits["test"])

splits = {'test': 'world_religions/test-00000-of-00001.parquet', 'validation': 'world_religions/validation-00000-of-00001.parquet', 'dev': 'world_religions/dev-00000-of-00001.parquet'}
World_Rel = pd.read_parquet("hf://datasets/cais/mmlu/" + splits["test"])

splits = {'train': 'main/train-00000-of-00001.parquet', 'test': 'main/test-00000-of-00001.parquet'}
GSM8K = pd.read_parquet("hf://datasets/openai/gsm8k/" + splits["test"])


##### Preprocessing #####
def preprocess_data(df):
    df = df.drop(columns=['subject'])
    df['answer'] = df['answer'] + 1
    df['gen_answer'] = None
    df['confidence'] = None
    df['correct'] = None
    return df

Col_Math = preprocess_data(Col_Math)
Biz_Ethics = preprocess_data(Biz_Ethics)
Prf_Law = preprocess_data(Prf_Law)
Com_Secu = preprocess_data(Com_Secu)
Anatomy = preprocess_data(Anatomy)
Astronomy = preprocess_data(Astronomy)
Marketing = preprocess_data(Marketing)
World_Rel = preprocess_data(World_Rel)

GSM8K['answer'] = GSM8K['answer'].str.extract(r'#### (\d+)')
GSM8K = GSM8K.dropna(subset=['answer'])
GSM8K['answer'] = GSM8K['answer'].astype(int)
GSM8K['gen_answer'] = None
GSM8K['confidence'] = None
GSM8K['correct'] = None


##### Vanilla prompts #####
from utils.phi3 import Phi3ChatCompletion
from utils.func import *
prompt = {
    "role": "system",
    "content": (
        "Read the question, provide your answer and your confidence in this answer. "
        "Note: The confidence indicates how likely you think your answer is true.\n"
        "Use the following format to answer:\n"
        "```Answer and Confidence (0-100): [ONLY the {answer_number}; not a complete sentence or symbols], "
        "[Your confidence level, please only include the numerical number in the range of 0-100]%```\n"
        "Only the answer and confidence, don't give me the explanations."
    )
}

def process_dataset(df, name):

    for index, row in tqdm(df.iterrows(), total=df.shape[0], desc=f"Processing {name} rows"):
        torch.cuda.empty_cache()
        context = retriever.get_relevant_documents(df.loc[index]['question'])
        #print(context[0])

        messages = [
            prompt,
            {"role": "user", "content": f"Answer the question based only on the following context: {context}\n"},
            {"role": "user", "content": (
                f"Question: {df.loc[index]['question']}\n"
                f"Options: 1. {df.loc[index]['choices'][0]}\n"
                f"2. {df.loc[index]['choices'][1]}\n"
                f"3. {df.loc[index]['choices'][2]}\n"
                f"4. {df.loc[index]['choices'][3]}\n"
                "Remember that you must have a format like '''Answer and Confidence (0-100): 3, 85%'''"
            )}
        ]
        
        gen_answer, confidence = extract_answer_and_confidence(Phi3ChatCompletion(messages))
        if gen_answer == 0 and confidence == 0:
            continue

        df.at[index, 'gen_answer'] = gen_answer
        df.at[index, 'confidence'] = confidence
        df.at[index, 'correct'] = 1 if gen_answer == row['answer'] else 0

    df.dropna(subset=['gen_answer'], inplace=True)
    print(df)

    # Save results to CSV
    #df.to_csv(f'./rag_results/{name}_RAG.csv', index=False)
    df.to_csv(f'./rag_wiki_results/{name}_RAG.csv', index=False)
    print(f"{name} dataset saved to {name}_RAG.csv")

process_dataset(Col_Math, "Col_Math")
process_dataset(Biz_Ethics, "Biz_Ethics")
process_dataset(Prf_Law, "Prf_Law")
process_dataset(Com_Secu, "Com_Secu")
process_dataset(Anatomy, "Anatomy")
process_dataset(Astronomy, "Astronomy")
process_dataset(Marketing, "Marketing")
process_dataset(World_Rel, "World_Rel")

for index, row in tqdm(GSM8K.iterrows(), total=GSM8K.shape[0], desc="Processing GSM8K rows"):
    context = retriever.get_relevant_documents(GSM8K.loc[index]['question'])

    messages = [
        prompt,
            {"role": "user", "content": f"Answer the question based only on the following context: {context}\n"},
            {"role": "user", "content": (
                f"Question: {GSM8K.loc[index]['question']}\n"
                "Remember that you must have a format like '''Answer and Confidence (0-100): 2, 85%'''"
            )}
    ]
    
    gen_answer, confidence = extract_answer_and_confidence(Phi3ChatCompletion(messages))
    if gen_answer == 0 and confidence == 0: continue
    
    GSM8K.at[index, 'gen_answer'] = gen_answer
    GSM8K.at[index, 'confidence'] = confidence
    GSM8K.at[index, 'correct'] = 1 if gen_answer == row['answer'] else 0

GSM8K = GSM8K.dropna()
print(GSM8K)

#GSM8K.to_csv('./rag_results/GSM8K_RAG.csv', index=False)
GSM8K.to_csv('./rag_wiki_results/GSM8K_RAG.csv', index=False)
print("GSM8K dataset saved to GSM8K_RAG.csv")
