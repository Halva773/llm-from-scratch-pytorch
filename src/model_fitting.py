from model.gpt import GPT
from model.bpe import BPE

if __name__ == "__main__":
    tokenizer = BPE(16000)
    tokenizer.load("savepoints/bpe16k.dill")

    
    