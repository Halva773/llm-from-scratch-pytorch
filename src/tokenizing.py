import pandas as pd
from bpe import BPE



if __name__ == '__main__':
    filepath = "../../dataset/poems.csv"
    data = pd.read_csv(filepath)
    data = data.dropna(subset=['text'])
    print(data.shape)
    print(data.head(2))

    bpe = BPE(vocab_size=16000)
    whole_text = "\n".join(data['text'])
    print("Text formatted. Going to fit tokenizer")
    bpe.fit(whole_text)
    bpe.save('../savepoints/bpe_16k.dill')
