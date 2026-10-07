import pandas as pd
from sklearn.model_selection import train_test_split

# Загружаем датасет
df = pd.read_csv('data/poems.csv')
print(f"Изначально: {len(df)} стихотворений, {df['writer'].nunique()} авторов")

# Убираем строки с пустым текстом
df = df.dropna(subset=['text']).copy()
print(f"После удаления пустых: {len(df)} стихотворений")

# Оставляем авторов с >= 30 стихами
min_poems = 30
author_counts = df['writer'].value_counts()
top_authors = author_counts[author_counts >= min_poems].index
df = df[df['writer'].isin(top_authors)].copy()
print(f"После фильтрации (>= {min_poems}): {len(df)} стихотворений, {df['writer'].nunique()} авторов")
print(f"Стихов у самого редкого автора: {df['writer'].value_counts().min()}")

# Разделяем на train (80%) и test (20%)
train_df, test_df = train_test_split(
    df,
    test_size=0.2,
    stratify=df['writer'],
    random_state=42
)

print(f"\nTrain: {len(train_df)}")
print(f"Test: {len(test_df)}")

# Сохраняем
train_df.to_csv('data/train.csv', index=False)
test_df.to_csv('data/test.csv', index=False)
print("\nГотово! Файлы data/train.csv и data/test.csv созданы.")