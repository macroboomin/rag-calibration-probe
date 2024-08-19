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

vectorstore = FAISS.load_local('./hotpot_qa/faiss', embeddings_model, allow_dangerous_deserialization=True)
#vectorstore = FAISS.load_local('./wiki/faiss_wiki', embeddings_model, allow_dangerous_deserialization=True)

retriever = vectorstore.as_retriever(
    search_type='mmr',
    search_kwargs={'k': 2, 'fetch_k': 30}
    #search_kwargs={'k': 5, 'fetch_k': 50}
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
from utils.func import *
from utils.class_probe import Probe
from transformers import AutoTokenizer, AutoModelForCausalLM, pipeline
from sklearn.preprocessing import StandardScaler

torch.random.manual_seed(202407) 
model = AutoModelForCausalLM.from_pretrained( 
    "microsoft/Phi-3-mini-4k-instruct",  
    device_map="cuda",  
    torch_dtype="auto",  
    trust_remote_code=True, 
    output_hidden_states=True, 
)

tokenizer = AutoTokenizer.from_pretrained("microsoft/Phi-3-mini-4k-instruct") 

pipe = pipeline( 
    "text-generation", 
    model=model, 
    tokenizer=tokenizer, 
) 

generation_args = { 
    "max_new_tokens": 30, 
    "return_full_text": False, 
    "temperature": 0.0, 
    "do_sample": False, 
} 

def Phi3ChatCompletion(messages):
    output = pipe(messages, **generation_args) 
    return output[0]['generated_text']

# Load probe model and tokenizer
probe = Probe(input_size=3072, hidden_sizes=[256, 128, 64], output_size=1).cuda()
probe.load_state_dict(torch.load('probe_model.pth'))
probe.eval()
print("Probe model loaded from probe_model.pth")

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
    model.eval()
    for index, row in tqdm(df.iterrows(), total=df.shape[0], desc=f"Processing {name} rows"):
        torch.cuda.empty_cache()
        context = retriever.get_relevant_documents(df.loc[index]['question'])

        if name == "GSM8K":
            messages = [
                prompt,
                {"role": "user", "content": f"Answer the question based only on the following context: {context}\n"},
                {"role": "user", "content": (
                    f"Question: {df.loc[index]['question']}\n"
                    "Remember that you must have a format like '''Answer and Confidence (0-100): 3, 85%'''"
                )}
            ]
        else :
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
        
        # Generate answer and confidence
        gen_answer, confidence = extract_answer_and_confidence(Phi3ChatCompletion(messages))
        if gen_answer == 0 and confidence == 0:
            continue

        # Probe model evaluation
        if name == "GSM8K":
            inputs = tokenizer(f"{df.loc[index]['question']}\n{gen_answer}", return_tensors="pt").input_ids.cuda()
        else:
            inputs = tokenizer(f"{df.loc[index]['question']}\n{df.loc[index]['choices'][gen_answer-1]}", return_tensors="pt").input_ids.cuda()
        
        with torch.no_grad():
            outputs = model(inputs, output_hidden_states=True)
            last_layer_features = outputs.hidden_states[-1][:, -1, :].detach().float()

        prob = probe(last_layer_features).squeeze().item()

        # Update the DataFrame
        df.at[index, 'gen_answer'] = gen_answer
        df.at[index, 'confidence'] = confidence
        df.at[index, 'probe_confidence'] = prob
        df.at[index, 'correct'] = 1 if gen_answer == row['answer'] else 0

    # Normalize the probe confidence score
    df = df.dropna(axis=0)
    min_val = df['probe_confidence'].min()
    max_val = df['probe_confidence'].max()
    df['probe_confidence'] = df['probe_confidence'].apply(lambda x: scaled_probe(x, min_val, max_val))

    print(df)

    df.to_csv(f'./rag_results/{name}_RAG_with_probe.csv', index=False)
    #df.to_csv(f'./rag_wiki_results/{name}_RAG_with_probe.csv', index=False)
    print(f"{name} dataset saved to {name}_RAG_with_probe.csv")

process_dataset(Col_Math, "Col_Math")
process_dataset(Biz_Ethics, "Biz_Ethics")
process_dataset(Prf_Law, "Prf_Law")
process_dataset(Com_Secu, "Com_Secu")
process_dataset(Anatomy, "Anatomy")
process_dataset(Astronomy, "Astronomy")
process_dataset(Marketing, "Marketing")
process_dataset(World_Rel, "World_Rel")
process_dataset(GSM8K, "GSM8K")
