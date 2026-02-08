# llm-from-scratch-pytorch
![Alt](https://repobeats.axiom.co/api/embed/6ca15cc238ef1f07307e0a19532198dee6c28f83.svg "Repobeats analytics image")

## Структура проекта

- `src/model/common/` — общие компоненты (BPE, attention, FFN, embeddings, dataset).
- `src/model/gpt1/model.py` — реализация модели `gpt1`.
- `src/model/gpt2/model.py` — реализация модели `gpt2`.
- `src/model/Llama/model.py` — реализация модели `llama`.
- `src/model_fitting.py` — обучение (точка входа для тренировки).
- `src/infer.py` — инференс/генерация (точка входа для текста).
- `src/tokenizing.py` — вспомогательные функции/скрипт для подготовки токенизации (если используется).
- `dataset/` — данные (например `dataset/poems.csv`).
- `savepoints/` — рекомендуемая папка для сохранений (чекпойнты и токенайзер).

## Данные

Скрипт обучения ожидает CSV `dataset/poems.csv` с колонкой `text`.

Пример датасета:
- `https://www.kaggle.com/datasets/grafstor/19-000-russian-poems`

## Установка зависимостей

- `pip install -r requirements.txt`

Примечание: если нужен GPU, `torch` лучше ставить по инструкции PyTorch под вашу CUDA, а не “как есть” из `requirements.txt`.

## Точки входа

- Обучение: `python src/model_fitting.py ...`
- Инференс: `python src/infer.py ...`

В обоих скриптах добавлен параметр выбора реализации модели:
- `--model_type gpt1|gpt2|llama`

## Обучение (model_fitting.py)

Важно: обучение может занимать очень много времени (особенно на CPU).

Примеры (запуск из корня репозитория):
- `python src/model_fitting.py --model_type gpt1 --device cuda --num_epoch 10 --save_dir savepoints --run_name gpt1_poems`
- `python src/model_fitting.py --model_type gpt2 --device cuda --num_epoch 10 --save_dir savepoints --run_name gpt2_poems`
- `python src/model_fitting.py --model_type llama --device cuda --num_epoch 10 --save_dir savepoints --run_name llama_poems`
- `python src/model_fitting.py --model_type gpt1 --device cpu --num_epoch 50 --save_dir savepoints --run_name gpt_poems --headAttention 8 --emb_size 512 --dict_size 2000 --dropout 0.1 --learning_rate 0.00001 --batch_size 128 --seq_len 64`
- `python src/model_fitting.py --model_type gpt2 --device cuda --layers 12 --headAttention 12 --emb_size 768 --dict_size 25000 --seq_len 256 --batch_size 16 --dropout 0.2 --learning_rate 2e-4 --num_epoch 10 --save_dir savepoints --run_name gpt2_poems_124m`
- `python src/model_fitting.py --model_type gpt2 --layers 12 --headAttention 12 --emb_size 768 --dict_size 25000 --seq_len 128 --batch_size 16 --learning_rate 1e-4 --dropout 0.2 --num_epoch 3`

Что сохраняется:
- токенайзер: `savepoints/bpe_<dict_size>.dill` (или `.json`, если нет `dill`)
- модель (чекпойнт): `savepoints/<run_name>.pth`

Если нужно без `dill`, токенайзер можно сохранять/грузить как JSON через `BPE.save_json(...)` / `BPE.load("...json")`.

## Инференс (infer.py)

Примеры:
- `python src/infer.py --model_type gpt1 --device cuda --model savepoints/gpt1_poems.pth --tokenizer savepoints/bpe_40000.dill --prompt "Привет, мир!" --max_new_tokens 50`
- `python src/infer.py --model_type gpt2 --device cuda --model savepoints/gpt2_poems.pth --tokenizer savepoints/bpe_40000.dill --prompt "Привет, мир!" --max_new_tokens 50 --do_sample --temperature 0.9 --top_k 50`
- `python src/infer.py --model_type llama --device cuda --model savepoints/llama_poems.pth --tokenizer savepoints/bpe_40000.dill --prompt "Привет, мир!" --max_new_tokens 50 --do_sample --temperature 0.9 --top_p 0.95 --top_k 50`
- `python src/infer.py --model_type gpt2 --device cuda --model savepoints/gpt2_poems.pth --tokenizer savepoints/bpe_25000.dill --prompt "…" --max_new_tokens 120 --do_sample --temperature 0.9 --top_p 0.95 --top_k 50`


Подсказка: `dict_size` токенайзера должен совпадать с `vocab_size` модели, иначе чекпойнт не загрузится.

## Docker (опционально)

CPU вариант:
- `docker build -t llm-from-scratch .`
- `docker run --rm -it -v ${PWD}/dataset:/app/dataset -v ${PWD}/savepoints:/app/savepoints llm-from-scratch python src/model_fitting.py --model_type gpt1 --device cpu --save_dir savepoints --run_name gpt_poems`

