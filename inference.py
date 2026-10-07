import pickle
import pandas as pd
from transformers import pipeline

# Загружаем модель
classifier = pipeline(
    "text-classification",
    model="./models/poems_author_classifier",
    tokenizer="./models/poems_author_classifier",
)

# Загружаем LabelEncoder
with open("./models/label_encoder.pkl", "rb") as f:
    le = pickle.load(f)


test_df = pd.read_csv('data/test.csv')
examples = test_df.sample(5, random_state=250)

print("=" * 70)
for i, (_, row) in enumerate(examples.iterrows(), 1):
    text = row['text']
    true_author = row['writer']

    result = classifier(text, truncation=True, max_length=128)[0]
    label_num = int(result['label'].split('_')[1])
    author_name = le.inverse_transform([label_num])[0]

    correct = "V" if author_name == true_author else "X"
    print(f"[{correct}] Стих {i}: {text[:80].strip()}...")
    print(f"    Настоящий автор:     {true_author}")
    print(f"    Предсказанный автор: {author_name}")
    print(f"    Уверенность:         {result['score']:.3f}")
    print("-" * 70)