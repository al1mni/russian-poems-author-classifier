# Классификация русских стихотворений по авторам

Дообучение русскоязычной языковой модели RuBERT для определения автора по тексту стихотворения.

## Данные

- Источник: [19 000 Russian Poems (Kaggle)](https://www.kaggle.com/datasets/grafstor/19-000-russian-poems)
- Объём: 19 316 стихотворений, 48 авторов
- После фильтрации: 45 авторов, 19 241 стихотворение

## Модель

- Базовая модель: `cointegrated/rubert-tiny2`
- Гиперпараметры: learning_rate=3e-5, batch_size=16, epochs=8, max_length=128
- Метрики на тесте: accuracy = [XX], F1 = [XX]

## Как запустить

1. `pip install -r requirements.txt`
2. `python prepare_data.py`
3. `python train.py`
4. `python inference.py`

## Ссылки

- Hugging Face: [ссылка]
- Датасет: https://www.kaggle.com/datasets/grafstor/19-000-russian-poems