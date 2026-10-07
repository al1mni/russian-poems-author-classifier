import pandas as pd
import pickle
from transformers import pipeline
from sklearn.metrics import accuracy_score

# Загружаем модель
classifier = pipeline(
    "text-classification",
    model="./models/poems_author_classifier",
    tokenizer="./models/poems_author_classifier",
)

# Загружаем LabelEncoder
with open("./models/label_encoder.pkl", "rb") as f:
    le = pickle.load(f)

# Берём 500 стихов из тестовой выборки
df = pd.read_csv('data/test.csv').sample(500, random_state=42)
print(f"Проверяем {len(df)} стихов...")

# Предсказания
preds = []
for text in df['text']:
    result = classifier(text, truncation=True, max_length=128)[0]
    label_num = int(result['label'].split('_')[1])
    author_name = le.inverse_transform([label_num])[0]
    preds.append(author_name)

# Точность
acc = accuracy_score(df['writer'], preds)
print(f"\nAccuracy на 500 стихах: {acc:.4f}")