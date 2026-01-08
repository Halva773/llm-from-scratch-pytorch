class BPE:
    def __init__(self, vocab_size):
        self.vocab_size = vocab_size
        
    def fit(self, text):
        self.tokens = []
        unique_symbols = self._get_unique_symbols(text)
        self.tokens.extend(unique_symbols)
        self._fitting(text)


    def _get_unique_symbols(self, text):
        unique_symbols = list(set(text))
        unique_symbols.sort()
        return unique_symbols[:self.vocab_size]
    
    def _fitting(self, text):
        body_text = list(text)
        pairs = self._count_pairs(body_text)

        for key, value in pairs.items():
            if value == max(pairs.values()):
                print(key, value)
        # print(pairs)

        # print(dict(sorted(pairs.items(), key=lambda x: x[1], reverse=True)))

    def _count_pairs(self, body_text: list):
        pairs = {}
        for i in range(len(body_text) - 1):
            pair = ''.join(body_text[i:i+2])
            pairs[pair] = pairs.get(pair, 0) + 1
        return pairs

    def _replace_most_popular_pair(self, body_text: list, pair: str):
        i = 0
        while i < len(body_text) - 1:
            current_pair = ''.join(body_text[i:i+2])
            print(current_pair)
            if pair == current_pair:
                body_text[i] = pair
                body_text.pop(i+1)
            i += 1
        return body_text



if __name__ == "__main__":
    bpe = BPE(vocab_size=1000)
    sample_text = "This is a sample text for BPE tokenization."
    bpe.fit(sample_text)
    # print(bpe.tokens)

