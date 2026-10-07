import pandas as pd
import numpy as np
import pickle
from datasets import Dataset, DatasetDict
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import accuracy_score, f1_score
from transformers import (
    AutoTokenizer,
    AutoModelForSequenceClassification,
    TrainingArguments,
    Trainer,
)

train_df = pd.read_csv('data/train.csv')
test_df = pd.read_csv('data/test.csv')
print(f"Train: {len(train_df)}, Test: {len(test_df)}")

le = LabelEncoder()
le.fit(train_df['writer'])
train_df['label'] = le.transform(train_df['writer'])
test_df['label'] = le.transform(test_df['writer'])
print(f"Классов: {len(le.classes_)}")
train_dataset = Dataset.from_pandas(train_df[['text', 'label']])
test_dataset = Dataset.from_pandas(test_df[['text', 'label']])
dataset = DatasetDict({"train": train_dataset, "test": test_dataset})

model_name = "cointegrated/rubert-tiny2"
tokenizer = AutoTokenizer.from_pretrained(model_name)

def tokenize_function(examples):
    return tokenizer(
        examples["text"],
        padding="max_length",
        truncation=True,
        max_length=128,
    )

tokenized_datasets = dataset.map(tokenize_function, batched=True)

# ========== 5. Модель ==========
model = AutoModelForSequenceClassification.from_pretrained(
    model_name,
    num_labels=len(le.classes_),
)

# ========== 6. Метрики ==========
def compute_metrics(eval_pred):
    predictions, labels = eval_pred
    predictions = np.argmax(predictions, axis=1)
    return {
        "accuracy": accuracy_score(labels, predictions),
        "f1": f1_score(labels, predictions, average="weighted"),
    }

training_args = TrainingArguments(
    output_dir="./results",
    eval_strategy="epoch",
    save_strategy="epoch",
    learning_rate=3e-5,
    per_device_train_batch_size=16,
    per_device_eval_batch_size=16,
    num_train_epochs=8,
    weight_decay=0.01,
    load_best_model_at_end=True,
    metric_for_best_model="accuracy",
    logging_steps=50,
)

# ========== 8. Обучение ==========
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=tokenized_datasets["train"],
    eval_dataset=tokenized_datasets["test"],
    compute_metrics=compute_metrics,
)

trainer.train()

# ========== 9. Сохранение ==========
model.save_pretrained("./models/poems_author_classifier")
tokenizer.save_pretrained("./models/poems_author_classifier")

with open("./models/label_encoder.pkl", "wb") as f:
    pickle.dump(le, f)

print("\nГотово! Модель сохранена в ./models/poems_author_classifier")
