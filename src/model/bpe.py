import heapq
import dill
from collections import defaultdict


class BPE:
    def __init__(self, vocab_size: int):
        self.vocab_size = vocab_size
        self.is_fitted = False

    # ============================================================
    # FIT
    # ============================================================

    def fit(self, text: str):
        # --- 1. Базовые символы ---
        base_symbols = sorted(set(text))
        self.base_vocab_size = len(base_symbols)

        if self.base_vocab_size >= self.vocab_size:
            base_symbols = base_symbols[: self.vocab_size]

        # token <-> id
        self.id2token = dict(enumerate(base_symbols))
        self.token2id = {t: i for i, t in self.id2token.items()}

        # --- 2. Текст -> ids ---
        ids = [self.token2id[c] for c in text]

        # --- 3. Структура "связанного списка" ---
        # prev[i] <-> next[i]
        prev = [-1] + list(range(len(ids) - 1))
        next = list(range(1, len(ids))) + [-1]

        # --- 4. Подсчёт пар и позиций ---
        pair_count = defaultdict(int)
        pair_pos = defaultdict(set)

        for i in range(len(ids) - 1):
            p = (ids[i], ids[i + 1])
            pair_count[p] += 1
            pair_pos[p].add(i)

        # max-heap по частоте
        heap = [(-cnt, pair) for pair, cnt in pair_count.items()]
        heapq.heapify(heap)

        merges = []
        next_token_id = len(self.token2id)

        max_merges = self.vocab_size - len(self.token2id)

        # ========================================================
        # MAIN LOOP
        # ========================================================
        for step in range(max_merges):
            # --- 5. Берём самую частую валидную пару ---
            while heap:
                neg_cnt, pair = heapq.heappop(heap)
                if pair_count[pair] == -neg_cnt and pair_count[pair] > 0:
                    break
            else:
                break  # больше нечего мёржить

            a, b = pair
            new_id = next_token_id
            next_token_id += 1

            # строковое представление токена — только здесь
            self.id2token[new_id] = self.id2token[a] + self.id2token[b]
            self.token2id[self.id2token[new_id]] = new_id
            merges.append((a, b))

            positions = list(pair_pos[pair])

            # --- 6. Применяем merge ТОЛЬКО в этих позициях ---
            for i in positions:
                j = next[i]
                if j == -1:
                    continue
                if ids[i] != a or ids[j] != b:
                    continue

                # соседи
                pi = prev[i]
                nj = next[j]

                # удаляем старые пары
                if pi != -1:
                    old = (ids[pi], ids[i])
                    pair_count[old] -= 1
                    pair_pos[old].discard(pi)

                if nj != -1:
                    old = (ids[j], ids[nj])
                    pair_count[old] -= 1
                    pair_pos[old].discard(j)

                # merge
                ids[i] = new_id
                next[i] = nj
                if nj != -1:
                    prev[nj] = i

                # добавляем новые пары
                if pi != -1:
                    new = (ids[pi], ids[i])
                    pair_count[new] += 1
                    pair_pos[new].add(pi)
                    heapq.heappush(heap, (-pair_count[new], new))

                if nj != -1:
                    new = (ids[i], ids[nj])
                    pair_count[new] += 1
                    pair_pos[new].add(i)
                    heapq.heappush(heap, (-pair_count[new], new))

            # пара больше не существует
            pair_count[pair] = 0
            pair_pos[pair].clear()

            if step % max(1, max_merges // 10) == 0:
                print(f"Обучено {step} merge")

        self.merges = merges
        self.is_fitted = True

    # ============================================================
    # ENCODE / DECODE
    # ============================================================

    def encode(self, text: str):
        if not self.is_fitted:
            raise RuntimeError("Call fit first")

        # greedy longest-match (как у тебя, но быстрее)
        if not hasattr(self, "_tokens_by_first_char"):
            d = {}
            for tok, tid in self.token2id.items():
                d.setdefault(tok[0], []).append(tok)
            for k in d:
                d[k].sort(key=lambda x: -len(x))
            self._tokens_by_first_char = d

        ids = []
        i = 0
        while i < len(text):
            ch = text[i]
            for tok in self._tokens_by_first_char.get(ch, []):
                if text.startswith(tok, i):
                    ids.append(self.token2id[tok])
                    i += len(tok)
                    break
            else:
                ids.append(self.token2id[ch])
                i += 1

        return ids

    def decode(self, token_ids):
        return "".join(self.id2token[i] for i in token_ids)

    # ============================================================
    # SAVE / LOAD
    # ============================================================

    def save(self, filename):
        with open(filename, "wb") as f:
            dill.dump(self, f)

    @classmethod
    def load(cls, filename):
        with open(filename, "rb") as f:
            return dill.load(f)
