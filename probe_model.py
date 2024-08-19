import pandas as pd
import re
import torch
import torch.nn as nn
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer, pipeline 
from sklearn.preprocessing import StandardScaler
from utils.func import *
from utils.class_probe import Probe 

import warnings
warnings.filterwarnings('ignore')

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

# Load datasets
animals_df = pd.read_csv('./data/animals_true_false.csv')
cities_df = pd.read_csv('./data/cities_true_false.csv')
companies_df = pd.read_csv('./data/companies_true_false.csv')
elements_df = pd.read_csv('./data/elements_true_false.csv')
facts_df = pd.read_csv('./data/facts_true_false.csv')
generated_df = pd.read_csv('./data/generated_true_false.csv')
inventions_df = pd.read_csv('./data/inventions_true_false.csv')

# Combine all datasets into one
combined_df = pd.concat([animals_df, cities_df, companies_df, elements_df, facts_df, generated_df, inventions_df])
combined_df['label'] = combined_df['label'].astype(int)

# Define the probe model architecture
hidden_sizes = [256, 128, 64]
probe = Probe(input_size=3072, hidden_sizes=hidden_sizes, output_size=1).cuda()
criterion = nn.BCELoss()
optimizer = torch.optim.Adam(probe.parameters(), lr=0.001)

# Function to train the probe model
def train_probe_model(dataset, model, tokenizer, probe, criterion, optimizer, scaler, num_epochs=5):
    model.eval()
    for epoch in range(num_epochs):
        total_loss = 0.0
        for index, row in tqdm(dataset.iterrows(), total=dataset.shape[0], desc=f"Training Epoch {epoch+1}"):
            statement = row['statement']
            label = row['label']
            
            with torch.no_grad():
                inputs = tokenizer(statement, return_tensors="pt").input_ids.cuda()
                outputs = model(inputs, output_hidden_states=True)
                last_layer_features = outputs.hidden_states[-1][:, -1, :].detach().float()  # Ensure float type

                # Normalize the features and ensure the dtype is float32
                last_layer_features = torch.tensor(scaler.fit_transform(last_layer_features.cpu().numpy()), dtype=torch.float32, device='cuda')
            
            target = torch.tensor([label], dtype=torch.float32).cuda()  # Ensure float type
            
            optimizer.zero_grad()
            prob = probe(last_layer_features).squeeze()  # Squeeze to match the target size
            prob = prob.unsqueeze(0) if prob.dim() == 0 else prob  # Ensure prob is at least 1D
            loss = criterion(prob, target)
            loss.backward()
            optimizer.step()
            
            total_loss += loss.item()
        print(f"Epoch {epoch+1}/{num_epochs}, Loss: {total_loss/len(dataset)}")

# Train the model with the combined dataset
scaler = StandardScaler()

train_probe_model(combined_df, model, tokenizer, probe, criterion, optimizer, scaler)

# Save model
torch.save(probe.state_dict(), 'probe_model.pth')
print("Probe model saved to probe_model.pth")
