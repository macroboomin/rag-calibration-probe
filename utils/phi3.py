import torch 
from transformers import AutoModelForCausalLM, AutoTokenizer, pipeline 

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