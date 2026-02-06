# llm-from-scratch-pytorch
![Alt](https://repobeats.axiom.co/api/embed/6ca15cc238ef1f07307e0a19532198dee6c28f83.svg "Repobeats analytics image")

## Datasets
- https://www.kaggle.com/datasets/grafstor/19-000-russian-poems (tokenizing)

## Быстрый запуск (сервер)

### 1) Зависимости

Установи зависимости:
- `pip install -r requirements.txt`

Примечание: если нужен GPU, установку `torch` лучше делать согласно инструкции PyTorch под вашу CUDA (а не “как есть” из `requirements.txt`).

### 2) Данные

Скрипт ожидает CSV `dataset/poems.csv` с колонкой `text`.

### 3) Обучение

Запуск из корня репозитория:
- `python src/model_fitting.py --device cuda --num_epoch 10 --save_dir savepoints --run_name gpt_poems`
- `python src/model_fitting.py --device cpu --num_epoch 50 --save_dir savepoints --run_name gpt_poems --headAttention 8 --emb_size 512 --dict_size 2000 --dropout 0.1 --learning_rate 0.00001 --batch_size 128 --seq_len 64`

После старта обучения сохраняются:
- токенайзер: `savepoints/bpe_<dict_size>.dill` (или `.json`, если нет `dill`)
- модель (чекпоинт): `savepoints/<run_name>.pth`

Если нужно без `dill`, токенайзер можно сохранять/грузить как JSON через `BPE.save_json(...)` / `BPE.load("...json")`.

### 4) Инференс (проверка, что веса грузятся)

- `python src/infer.py --device cuda --model savepoints/gpt_poems.pth --tokenizer savepoints/bpe_40000.dill --prompt "Привет, мир!" --max_new_tokens 50`

## Docker (опционально)

CPU вариант:
- `docker build -t llm-from-scratch .`
- `docker run --rm -it -v ${PWD}/dataset:/app/dataset -v ${PWD}/savepoints:/app/savepoints llm-from-scratch python src/model_fitting.py --device cpu --save_dir savepoints --run_name gpt_poems`
