# =========================
# 1. 라이브러리 불러오기
# =========================
import pandas as pd
import torch

from datasets import Dataset

from transformers import (
    AutoTokenizer,
    AutoModelForSequenceClassification,
    TrainingArguments,
    Trainer
)

from sklearn.metrics import accuracy_score


# =========================
# 2. NSMC 데이터셋 불러오기
# =========================
train = pd.read_csv(
    "ratings_train.txt",
    sep="\t"
)

test = pd.read_csv(
    "ratings_test.txt",
    sep="\t"
)

# 결측치 제거
train = train.dropna()
test = test.dropna()

print(train.head())


# =========================
# 3. 필요한 컬럼만 사용
# =========================
train = train[["document", "label"]]
test = test[["document", "label"]]


# =========================
# 4. HuggingFace Dataset 변환
# =========================
train_dataset = Dataset.from_pandas(train)
test_dataset = Dataset.from_pandas(test)


# =========================
# 5. KoELECTRA 토크나이저 불러오기
# =========================
MODEL_NAME = "monologg/koelectra-base-v3-discriminator"

tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)


# =========================
# 6. 토크나이징 함수
# =========================
def tokenize_function(examples):
    return tokenizer(
        examples["document"],
        padding="max_length",
        truncation=True,
        max_length=128
    )


# =========================
# 7. 데이터 토크나이징
# =========================
tokenized_train = train_dataset.map(
    tokenize_function,
    batched=True
)

tokenized_test = test_dataset.map(
    tokenize_function,
    batched=True
)


# =========================
# 8. PyTorch 형식 변환
# =========================
tokenized_train.set_format(
    type="torch",
    columns=[
        "input_ids",
        "attention_mask",
        "label"
    ]
)

tokenized_test.set_format(
    type="torch",
    columns=[
        "input_ids",
        "attention_mask",
        "label"
    ]
)


# =========================
# 9. 모델 불러오기
# =========================
model = AutoModelForSequenceClassification.from_pretrained(
    MODEL_NAME,
    num_labels=2
)


# =========================
# 10. 평가 함수
# =========================
def compute_metrics(eval_pred):
    logits, labels = eval_pred

    predictions = torch.argmax(
        torch.tensor(logits),
        dim=-1
    )

    acc = accuracy_score(
        labels,
        predictions
    )

    return {"accuracy": acc}


# =========================
# 11. 학습 설정
# =========================
training_args = TrainingArguments(
    output_dir="./results",

    # 학습 횟수
    num_train_epochs=1,

    # 배치 크기
    per_device_train_batch_size=8,
    per_device_eval_batch_size=8,

    # 로그 출력
    logging_steps=100,

    report_to="none"
)


# =========================
# 12. Trainer 생성
# =========================
trainer = Trainer(
    model=model,
    args=training_args,

    # 데이터 일부만 사용 (속도 개선)
    train_dataset=tokenized_train.select(range(500)),
    eval_dataset=tokenized_test.select(range(100)),

    compute_metrics=compute_metrics
)


# =========================
# 13. 모델 학습
# =========================
trainer.train()


# =========================
# 14. 모델 평가
# =========================
result = trainer.evaluate()

print(result)


# =========================
# 15. 직접 예측
# =========================
text = "진짜 재미있고 감동적인 영화였다"

inputs = tokenizer(
    text,
    return_tensors="pt",
    truncation=True,
    padding=True,
    max_length=128
)

with torch.no_grad():
    outputs = model(**inputs)

prediction = torch.argmax(
    outputs.logits,
    dim=-1
).item()

if prediction == 1:
    print("긍정 리뷰 😊")
else:
    print("부정 리뷰 😢")