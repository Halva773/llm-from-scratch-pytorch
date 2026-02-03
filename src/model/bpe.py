import heapq
import sys
import json
try:
    import dill
except ModuleNotFoundError:  # pragma: no cover
    dill = None
try:
    from tqdm import tqdm
except ModuleNotFoundError:  # pragma: no cover
    def tqdm(x, **kwargs):
        return x
from collections import defaultdict
from pathlib import Path

# Backward-compat for older dill files that referenced module name `bpe`.
# If `BPE` was pickled as `bpe.BPE`, unpickling will try to import `bpe`.
sys.modules.setdefault("bpe", sys.modules[__name__])


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
        for step in tqdm(range(max_merges), desc="Fitting BPE", unit="merge"):
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

        self.merges = merges
        self.is_fitted = True

    # ============================================================
    # ENCODE / DECODE
    # ============================================================

    def encode(self, text: str):
        if not self.is_fitted:
            raise RuntimeError("Tokenizer is not fitted. Call fit() or load().")

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

    def to_dict(self) -> dict:
        if not self.is_fitted:
            raise RuntimeError("Tokenizer is not fitted. Call fit() first.")

        max_id = max(self.id2token.keys()) if self.id2token else -1
        id2token_list = [self.id2token[i] for i in range(max_id + 1)]

        return {
            "vocab_size": self.vocab_size,
            "is_fitted": self.is_fitted,
            "base_vocab_size": getattr(self, "base_vocab_size", None),
            "id2token": id2token_list,
            "merges": [list(p) for p in getattr(self, "merges", [])],
        }

    @classmethod
    def from_dict(cls, data: dict):
        obj = cls(vocab_size=int(data["vocab_size"]))
        obj.is_fitted = bool(data.get("is_fitted", True))

        id2token_list = data["id2token"]
        obj.id2token = {i: tok for i, tok in enumerate(id2token_list)}
        obj.token2id = {tok: i for i, tok in obj.id2token.items()}
        obj.base_vocab_size = data.get("base_vocab_size", len(obj.token2id))
        obj.merges = [tuple(p) for p in data.get("merges", [])]
        return obj

    def save_json(self, filename):
        path = Path(filename)
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("w", encoding="utf-8") as f:
            json.dump(self.to_dict(), f, ensure_ascii=False)

    def save(self, filename):
        if dill is None:
            raise ModuleNotFoundError("dill is required to save tokenizer as .dill. Install dill or use save_json().")
        path = Path(filename)
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("wb") as f:
            dill.dump(self, f)

    @classmethod
    def load(cls, filename):
        path = Path(filename)
        if path.suffix.lower() == ".json":
            with path.open("r", encoding="utf-8") as f:
                return cls.from_dict(json.load(f))
        if dill is None:
            raise ModuleNotFoundError("dill is required to load tokenizer from .dill. Install dill or use a .json tokenizer.")
        with path.open("rb") as f:
            obj = dill.load(f)
        obj.is_fitted = True
        return obj
