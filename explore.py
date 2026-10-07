import pandas as pd

# Загружаем датасет из папки data
df = pd.read_csv('data/poems.csv')

print("Первые 5 строк:")
print(df.head())

print("\nРазмер датасета (строки, столбцы):")
print(df.shape)

print("\nСтолбцы:")
print(df.columns.tolist())

print("\nУникальных авторов:", df['writer'].nunique())

print("\nТоп-20 авторов по количеству стихов:")
print(df['writer'].value_counts().head(20))