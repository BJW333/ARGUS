import os
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
#MY_TOKEN = "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"
#os.environ["HF_AUTH_TOKEN"] = MY_TOKEN
MODEL_CHECKPOINT = "bjw333/macosargus-code"
os.environ["TOKENIZERS_PARALLELISM"] = "false"


#global vars for tokenizer and model so its in cache and can use this file as a code gen method etc
argus_tokenizer = None
argus_model = None

def load_model():
    global argus_tokenizer, argus_model, device
    if argus_tokenizer is None or argus_model is None:
        print("loading model and tokenizer from Hugging Face")
        #inputs must be on the same device as the model or error will occur 
        if torch.backends.mps.is_available():
            device = "mps"
        elif torch.cuda.is_available():
            device = "cuda"
        else:
            device = "cpu"
            
        print(f"Using device: {device}")
    
        argus_tokenizer = AutoTokenizer.from_pretrained(
            MODEL_CHECKPOINT,
            #token=MY_TOKEN,
            trust_remote_code=True
        )
        
        argus_model = AutoModelForCausalLM.from_pretrained(
            MODEL_CHECKPOINT,
            #token=MY_TOKEN,
            torch_dtype=torch.float16,  #change or remove for FP32 if desired
            device_map="auto",
            trust_remote_code=True
        )
        
        #aprently this line below is meant for when the model is being used to generate code
        argus_model.eval()
        
        print("model and tokenizer loaded")
        
    return argus_tokenizer, argus_model, device


def argus_code_generation(prompt, max_length=1024):
    """
    generate code based on the given prompt using the loaded model
    
    prompt (str): The input prompt for code generation.
    max_length (int): The maximum number of tokens to generate (default is 1024 this value can be lowered to 512 or 128 whatver is needed).

    returns
       generated_text (str): The generated code as a string.
    """
    #added prompt + newline so that the prompt and generated code arent all printed out on one line 
    #prompt = prompt + "\n"
    
    tokenizer, model, device = load_model()
    inputs = tokenizer(prompt, return_tensors="pt")
    
    #device var getting loaded globally with load model function 
    #print(f"Using device: {device}")
    
    inputs = {k: v.to(device) for k, v in inputs.items()}
    #also applies additional optimizations that can reduce memory usage and improve performance during model evaluation. that is what the below line does
    with torch.inference_mode():
        outputs = model.generate(**inputs, max_length=max_length)
        
    generated_text = tokenizer.decode(outputs[0], skip_special_tokens=True)
    return generated_text.strip()
    #return generated_text


#testing purposes
#if __name__ == "__main__":
#    prompt = "Write a Python function for a program figuring out if a array has even and odd number"
#    #prompt = prompt + "\n"
#    code = argus_code_generation(prompt, max_length=1024)
#    print(code)
