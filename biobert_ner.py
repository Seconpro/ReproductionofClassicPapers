import torch
import numpy as np
import pandas as pd
from datasets import Dataset, DatasetDict
from transformers import AutoTokenizer, AutoModelForTokenClassification, TrainingArguments, Trainer
from transformers import DataCollatorForTokenClassification
from seqeval.metrics import precision_score, recall_score, f1_score, accuracy_score

# ==================== 标签定义 ====================
# NCBI Disease 数据集只有三种标签
label_list = ['O', 'B-Disease', 'I-Disease']
label2id = {label: i for i, label in enumerate(label_list)}
id2label = {i: label for i, label in enumerate(label_list)}
num_labels = len(label_list)


# ==================== 读取 CoNLL 格式的 TSV 文件 ====================
def read_conll_file(file_path):
    sentences = []
    current_tokens = []
    current_tags = []

    with open(file_path, 'r', encoding='utf-8') as f:
        for line in f:
            line = line.strip()
            if line == '':
                if current_tokens:
                    sentences.append({
                        'tokens': current_tokens,  # 直接存列表
                        'ner_tags': [label2id[tag] for tag in current_tags]
                    })
                    current_tokens = []
                    current_tags = []
            else:
                parts = line.split('\t')
                if len(parts) >= 2:
                    token = parts[0]
                    tag = parts[-1]
                    current_tokens.append(token)
                    current_tags.append(tag)
        if current_tokens:
            sentences.append({
                'tokens': current_tokens,
                'ner_tags': [label2id[tag] for tag in current_tags]
            })

    df = pd.DataFrame(sentences)
    return df


# ==================== 1. 加载本地数据 ====================
print("正在加载本地数据...")
train_df = read_conll_file(r"C:\Users\aa\Desktop\工作学习内容\现阶段任务\训练数据集\NCBI\train.tsv")
val_df = read_conll_file(r"C:\Users\aa\Desktop\工作学习内容\现阶段任务\训练数据集\NCBI\devel.tsv")
test_df = read_conll_file(r"C:\Users\aa\Desktop\工作学习内容\现阶段任务\训练数据集\NCBI\test.tsv")

print(f"训练集样本数: {len(train_df)}")
print(f"验证集样本数: {len(val_df)}")
print(f"测试集样本数: {len(test_df)}")

# 构建 HuggingFace DatasetDict
dataset = DatasetDict({
    "train": Dataset.from_pandas(train_df),
    "validation": Dataset.from_pandas(val_df),
    "test": Dataset.from_pandas(test_df),
})

# ==================== 2. 加载模型和分词器 ====================
print("加载 BioBERT 模型和分词器...")
model_checkpoint = "dmis-lab/biobert-base-cased-v1.1"

# 使用 AutoTokenizer（自动选择快速分词器，兼容 word_ids）
tokenizer = AutoTokenizer.from_pretrained(model_checkpoint)

model = AutoModelForTokenClassification.from_pretrained(
    model_checkpoint,
    num_labels=num_labels,
    id2label=id2label,
    label2id=label2id,
)

# ==================== 3. 数据预处理 ====================
max_length = 128


def tokenize_and_align_labels(examples):
    tokenized_inputs = tokenizer(
        examples["tokens"], truncation=True, is_split_into_words=True,
        max_length=max_length, padding="max_length"
    )
    word_ids = tokenized_inputs.word_ids()  # 现在应该可用，因为是快速分词器
    label = examples["ner_tags"]
    previous_word_idx = None
    label_ids = []
    for word_idx in word_ids:
        if word_idx is None:
            label_ids.append(-100)
        elif word_idx != previous_word_idx:
            label_ids.append(label[word_idx])
        else:
            label_ids.append(-100)
        previous_word_idx = word_idx
    tokenized_inputs["labels"] = label_ids
    return tokenized_inputs


print("数据预处理中...")
tokenized_datasets = dataset.map(tokenize_and_align_labels, batched=False)

# ==================== 4. 数据整理器 ====================
data_collator = DataCollatorForTokenClassification(tokenizer)


# ==================== 5. 评估指标 ====================
def compute_metrics(eval_pred):
    predictions, labels = eval_pred
    predictions = np.argmax(predictions, axis=2)

    # 去除 -100 的标签，并转换为可读字符串
    true_predictions = [
        [id2label[p] for (p, l) in zip(prediction, label) if l != -100]
        for prediction, label in zip(predictions, labels)
    ]
    true_labels = [
        [id2label[l] for (p, l) in zip(prediction, label) if l != -100]
        for prediction, label in zip(predictions, labels)
    ]

    return {
        "precision": precision_score(true_labels, true_predictions),
        "recall": recall_score(true_labels, true_predictions),
        "f1": f1_score(true_labels, true_predictions),
        "accuracy": accuracy_score(true_labels, true_predictions),
    }


# ==================== 6. 训练参数 ====================
training_args = TrainingArguments(
    output_dir="./biobert-ner",
    evaluation_strategy="epoch",
    save_strategy="epoch",
    learning_rate=3e-5,                # 改为 3e-5
    per_device_train_batch_size=16,
    per_device_eval_batch_size=16,
    num_train_epochs=10,               # 改为 10
    weight_decay=0.01,
    warmup_ratio=0.1,                  # 新增 warmup
    load_best_model_at_end=True,
    metric_for_best_model="f1",
    report_to="none",
)

# ==================== 7. 创建 Trainer ====================
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=tokenized_datasets["train"],
    eval_dataset=tokenized_datasets["validation"],
    data_collator=data_collator,
    compute_metrics=compute_metrics,
)

# ==================== 8. 开始训练 ====================
print("开始训练...")
trainer.train()

# ==================== 9. 在测试集上评估 ====================
print("在测试集上评估...")
test_results = trainer.evaluate(tokenized_datasets["test"])
print("测试集结果：", test_results)

# ==================== 10. 保存最终模型 ====================
trainer.save_model("./biobert-ner-final")
print("模型已保存到 ./biobert-ner-final")