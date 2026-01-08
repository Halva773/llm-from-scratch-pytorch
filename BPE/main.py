class BPE:
    def __init__(self, vocab_size):
        self.vocab_size = vocab_size
        
    def fit(self, text):
        self.tokens = []
        unique_symbols = self._get_unique_symbols(text)
        self.tokens.extend(unique_symbols)
        self._fitting(text)
        self.__create_id2token()
        self.__create_token2id()

    def _get_unique_symbols(self, text):
        unique_symbols = list(set(text))
        unique_symbols.sort()
        return unique_symbols[:self.vocab_size]
    
    def _fitting(self, text):
        body_text = list(text)

        while (len(self.tokens) < self.vocab_size) and (len(body_text) != 1):
            pairs = self._count_pairs(body_text)

            popular_pairs = set()
            for key, value in pairs.items():
                if value == max(pairs.values()):
                    popular_pairs.add(key)
            body_text, pair = self._replace_most_popular_pair(body_text, pairs=popular_pairs)
            self.tokens.append(pair)
        
    def _count_pairs(self, body_text: list):
        pairs = {}
        for i in range(len(body_text) - 1):
            pair = ''.join(body_text[i:i+2])
            pairs[pair] = pairs.get(pair, 0) + 1
        return pairs

    def _replace_most_popular_pair(self, body_text: list, pairs: set):
        i = 0
        pair = None
        while i < len(body_text) - 1:
            current_pair = ''.join(body_text[i:i+2])
            if (pair is None) and (current_pair in pairs):
                pair = current_pair
                body_text[i] = pair
                body_text.pop(i+1)
                break
            i += 1
        while i < len(body_text) - 1:
            current_pair = ''.join(body_text[i:i+2])
            if pair == current_pair:
                body_text[i] = pair
                body_text.pop(i+1)
            i += 1
        return body_text, pair
    
    def __create_id2token(self):
        self.id2token = {i: token for i, token in enumerate(self.tokens)}

    def __create_token2id(self):
        self.token2id = {token: i for i, token in enumerate(self.tokens)}



if __name__ == "__main__":


    
    bpe = BPE(vocab_size=30)
    sample_text = "Из кузова в кузов шла перегрузка арбузов. В грозу в грязи от груза арбузов развалился кузов."
    bpe.fit(sample_text)
    assert bpe.tokens == [' ','.','В','И','а','б','в','г','е','з','и','к','л','о','п','р','с','т','у','ш','я','уз','узо','узов','а ','гр',' к',' кузов',' гр','а а']


    bpe = BPE(vocab_size=31)
    sample_text = "Однажды был случай в далёком Макао: макака коалу в какао макала, коала лениво какао лакала, макака макала, коала икала."
    bpe.fit(sample_text)
    assert bpe.tokens == [' ',',','.',':','М','О','а','б','в','д','е','ж','и','й','к','л','м','н','о','с','у','ч','ы','ё','ка','ла','ака','ко',' м',' мака',' ко']
    print("All tests passed.")

