# =========================
# 1. 라이브러리 불러오기
# =========================
import time
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
train = pd.read_csv("ratings_train.txt", sep="\t")
test = pd.read_csv("ratings_test.txt", sep="\t")

train = train.dropna()
test = test.dropna()

train = train[["document", "label"]]
test = test[["document", "label"]]

train_dataset = Dataset.from_pandas(train)
test_dataset = Dataset.from_pandas(test)


# =========================
# 3. 비교할 모델 목록
# =========================
MODELS = [
    "monologg/koelectra-base-v3-discriminator",
    "answerdotai/ModernBERT-base",
    "klue/roberta-base",
    "snunlp/KR-ELECTRA-discriminator",
]

SAMPLE_TEXT = "진짜 재미있고 감동적인 영화였다"
TRAIN_SAMPLES = 500
EVAL_SAMPLES  = 100


# =========================
# 4. 평가 함수
# =========================
def compute_metrics(eval_pred):
    logits, labels = eval_pred
    predictions = torch.argmax(torch.tensor(logits), dim=-1)
    return {"accuracy": accuracy_score(labels, predictions)}


# =========================
# 5. 모델별 실행 및 시간 측정
# =========================
results = []

for model_name in MODELS:
    print(f"\n{'='*60}")
    print(f"모델: {model_name}")
    print(f"{'='*60}")
    timing = {"model": model_name}

    # --- 토크나이저 & 모델 로딩 ---
    t0 = time.time()
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForSequenceClassification.from_pretrained(
        model_name, num_labels=2
    )
    timing["load_sec"] = round(time.time() - t0, 2)
    print(f"[로딩]    {timing['load_sec']}s")

    # --- 토크나이징 ---
    def tokenize_function(examples):
        return tokenizer(
            examples["document"],
            padding="max_length",
            truncation=True,
            max_length=128
        )

    t0 = time.time()
    tokenized_train = train_dataset.map(tokenize_function, batched=True)
    tokenized_test  = test_dataset.map(tokenize_function,  batched=True)
    timing["tokenize_sec"] = round(time.time() - t0, 2)
    print(f"[토크나이징] {timing['tokenize_sec']}s")

    for ds in (tokenized_train, tokenized_test):
        ds.set_format(
            type="torch",
            columns=["input_ids", "attention_mask", "label"]
        )

    # --- 학습 ---
    training_args = TrainingArguments(
        output_dir=f"./results/{model_name.replace('/', '_')}",
        num_train_epochs=1,
        per_device_train_batch_size=8,
        per_device_eval_batch_size=8,
        logging_steps=100,
        report_to="none"
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=tokenized_train.select(range(TRAIN_SAMPLES)),
        eval_dataset=tokenized_test.select(range(EVAL_SAMPLES)),
        compute_metrics=compute_metrics
    )

    t0 = time.time()
    trainer.train()
    timing["train_sec"] = round(time.time() - t0, 2)
    print(f"[학습]    {timing['train_sec']}s")

    # --- 평가 ---
    t0 = time.time()
    eval_result = trainer.evaluate()
    timing["eval_sec"] = round(time.time() - t0, 2)
    timing["accuracy"] = round(eval_result.get("eval_accuracy", 0), 4)
    print(f"[평가]    {timing['eval_sec']}s  |  accuracy: {timing['accuracy']}")

    # --- 단일 추론 ---
    inputs = tokenizer(
        SAMPLE_TEXT,
        return_tensors="pt",
        truncation=True,
        padding=True,
        max_length=128
    )

    t0 = time.time()
    with torch.no_grad():
        outputs = model(**inputs)
    timing["inference_sec"] = round(time.time() - t0, 4)

    prediction = torch.argmax(outputs.logits, dim=-1).item()
    timing["prediction"] = "긍정" if prediction == 1 else "부정"
    print(f"[추론]    {timing['inference_sec']}s  |  결과: {timing['prediction']}")

    timing["total_sec"] = round(
        timing["load_sec"] + timing["tokenize_sec"] +
        timing["train_sec"] + timing["eval_sec"],
        2
    )
    results.append(timing)

    # 메모리 해제
    del model, tokenizer, trainer
    torch.cuda.empty_cache() if torch.cuda.is_available() else None


# =========================
# 6. 결과 비교표 출력
# =========================
print(f"\n{'='*60}")
print("모델 실행시간 비교")
print(f"{'='*60}")

df = pd.DataFrame(results).set_index("model")
df.index.name = "모델"
df.columns = ["로딩(s)", "토크나이징(s)", "학습(s)", "평가(s)",
              "정확도", "추론(s)", "예측", "합계(s)"]
print(df.to_string())
