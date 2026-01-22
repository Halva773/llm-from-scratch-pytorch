import dill


class BPE:
    def __init__(self, vocab_size: int):
        self.vocab_size = vocab_size
        self.merges = []
        self.is_fitted = False

    def fit(self, text: str):
        self.unique_symbols = self._get_unique_symbols(text)

        # В vocab_size входят и базовые символы, и merge-токены
        max_merges = self.vocab_size - len(self.unique_symbols)
        if max_merges < 0:
            self.unique_symbols = self.unique_symbols[: self.vocab_size]
            max_merges = 0

        self._fitting(text, max_merges=max_merges)

        self.tokens = self._build_vocab_from_merges()
        self.__create_id2token()
        self.__create_token2id()

        # Сброс кэша индексов для encode
        if hasattr(self, "_tokens_by_first_char"):
            delattr(self, "_tokens_by_first_char")

        self.is_fitted = True

    def _get_unique_symbols(self, text: str):
        return sorted(set(text))

    def _fitting(self, text: str, max_merges: int):
        body_text = list(text)

        while (len(self.merges) < max_merges) and (len(body_text) > 1):
            pairs = self._count_pairs(body_text)
            pair = max(pairs, key=pairs.get)
            body_text, pair = self._replace_most_popular_pair(body_text, pair=pair)
            self.merges.append(pair)

    def _count_pairs(self, body_text: list):
        pairs = {}
        for i in range(len(body_text) - 1):
            pair = (body_text[i], body_text[i + 1])
            pairs[pair] = pairs.get(pair, 0) + 1
        return pairs

    # ВАЖНО: заменяем ВСЕ вхождения пары, а не первое
    def _replace_most_popular_pair(self, body_text: list, pair: tuple):
        i = 0
        while i < len(body_text) - 1:
            if (body_text[i], body_text[i + 1]) == pair:
                body_text[i] = body_text[i] + body_text[i + 1]
                body_text.pop(i + 1)
                # i не увеличиваем — после склейки могут образоваться новые совпадения на той же позиции
            else:
                i += 1
        return body_text, pair

    # Детерминированный порядок токенов: сначала базовые символы, затем merge'и по порядку обучения
    def _build_vocab_from_merges(self):
        vocab = list(self.unique_symbols)
        seen = set(vocab)

        for a, b in self.merges:
            tok = a + b
            if tok not in seen:
                vocab.append(tok)
                seen.add(tok)

        return vocab[: self.vocab_size]

    def __create_id2token(self):
        self.id2token = {i: token for i, token in enumerate(self.tokens)}

    def __create_token2id(self):
        self.token2id = {token: i for i, token in enumerate(self.tokens)}

    # Быстрый жадный энкодинг по условию
    def encode(self, text: str):
        if not self.is_fitted:
            raise RuntimeError("Call fit first")
        if not text:
            return []

        # Индекс: первый символ -> токены, отсортированные по (длина desc, id asc)
        if not hasattr(self, "_tokens_by_first_char"):
            d = {}
            for tok in self.tokens:
                d.setdefault(tok[0], []).append(tok)
            for ch in d:
                d[ch].sort(key=lambda t: (-len(t), self.token2id[t]))
            self._tokens_by_first_char = d

        ids = []
        i = 0
        n = len(text)

        while i < n:
            ch = text[i]
            candidates = self._tokens_by_first_char.get(ch, [])

            chosen = None
            for tok in candidates:
                if text.startswith(tok, i):
                    chosen = tok
                    break

            if chosen is None:
                chosen = ch

            ids.append(self.token2id[chosen])
            i += len(chosen)

        return ids
    
    def decode(self, token_ids: list[int]):
        return "".join([self.id2token[i] for i in token_ids])
    
    def save(self, filename):
        with open(filename, 'wb') as f:
            dill.dump(self, f)
        print(f"Объект сохранён в {filename}")


    @classmethod
    def load(cls, filename):
        with open(filename, 'rb') as f:
            obj = dill.load(f)
                
        print(f"Объект загружен из {filename}")
        return obj
    


if __name__ == "__main__":
    bpe = BPE(vocab_size=30)
    sample_text = "This is a sample text for BPE tokenization."
    bpe.fit(sample_text)
    tokens = bpe.encode('sample text')
    print(tokens)

    bpe.save('data/bpe.dill')

    loaded_bpe = BPE.load('data/bpe.dill')
    decoded = loaded_bpe.decode(tokens)
    print(decoded)