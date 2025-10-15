import os, glob, logging, torch, re
from typing import List, Dict, Tuple
from datasets import Dataset
from transformers import (
    AutoTokenizer, AutoModelForTokenClassification,
    TrainingArguments, Trainer
)
import matplotlib.pyplot as plt
import numpy as np

BASE_DIR = r"Final_v1\AIFORTHAI-LST20Corpus\LST20_Corpus_final"
ENCODING = "utf-8"
SAVE_DIR = "./ner_modelfinal_25"

logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
logger = logging.getLogger(__name__)

# --- Load files ---
def load_split_lines(base_dir: str, split: str) -> List[Tuple[str, str]]:
    pattern = os.path.join(base_dir, split, "*.txt")
    files = sorted(glob.glob(pattern))
    if not files:
        raise FileNotFoundError(f"ไม่พบไฟล์ใน {pattern}")
    logger.info(f"{split}: พบ {len(files)} ไฟล์")
    lines = []
    for fp in files:
        with open(fp, encoding=ENCODING) as f:
            for line in f:
                lines.append((fp, line))
            lines.append((fp, "\n"))
    return lines

# --- Parse to sentences ---
def parse_lines_to_sentences(raw: List[Tuple[str, str]]) -> List[List[List[str]]]:
    sentences, current = [], []
    for i, (fname, line) in enumerate(raw):
        line = line.strip()
        if not line:
            if current:
                sentences.append(current)
                current = []
            continue
        parts = re.split(r"\s+", line)
        if len(parts) == 4:
            current.append(parts)
        elif len(parts) > 4:
            current.append(parts[:4])
            logger.warning(f"[{os.path.basename(fname)}] ซ่อมบรรทัด {i}: {line} -> {parts[:4]}")
        else:
            logger.warning(f"[{os.path.basename(fname)}] ข้ามบรรทัดผิดรูปแบบ ({i}): {line}")

    if current:
        sentences.append(current)
    logger.info(f"รวม {len(sentences)} ประโยค (ยาวสุด {max(map(len, sentences))} token)")
    return sentences

# --- Label extraction ---
def extract_labels(sentences: List[List[List[str]]]) -> List[str]:
    labels = sorted({f"{tok[1]}|{tok[2]}|{tok[3]}" for sent in sentences for tok in sent})
    logger.info(f"พบ label รวม {len(labels)} ชนิด")
    return labels

# --- Convert to HF dataset ---
def convert_to_hf(sentences: List[List[List[str]]], label2id: Dict[str, int]) -> List[Dict]:
    return [
        {"tokens": [w[0] for w in sent],
         "labels": [label2id[f"{w[1]}|{w[2]}|{w[3]}"] for w in sent]}
        for sent in sentences
    ]

# --- Token alignment ---
def tokenize_and_align(batch: Dict) -> Dict:
    tok_out = tokenizer(
        batch["tokens"],
        is_split_into_words=True,
        truncation=True,
        padding="max_length",
        max_length=510,
        return_offsets_mapping=False
    )

    aligned_labels = []
    for i, word_ids in enumerate(tok_out.word_ids(batch_index=j) for j in range(len(batch["tokens"]))):
        sent_labels, prev = [], None
        for idx in word_ids:
            if idx is None:
                sent_labels.append(-100)
            elif idx != prev:
                sent_labels.append(batch["labels"][i][idx])
            else:
                sent_labels.append(-100)
            prev = idx
        aligned_labels.append(sent_labels)

    tok_out["labels"] = aligned_labels
    return tok_out

# --- Compute metrics ---
def compute_metrics(p):
    predictions, labels = p
    predictions = np.argmax(predictions, axis=2)

    true_labels = [
        [id2label[l] for (p, l) in zip(pred, label) if l != -100]
        for pred, label in zip(predictions, labels)
    ]
    true_predictions = [
        [id2label[p] for (p, l) in zip(pred, label) if l != -100]
        for pred, label in zip(predictions, labels)
    ]

    correct, total = 0, 0
    for t, p in zip(true_labels, true_predictions):
        for a, b in zip(t, p):
            total += 1
            if a == b:
                correct += 1
    acc = correct / total if total > 0 else 0.0
    return {"accuracy": acc}

# --- MAIN ---
if __name__ == "__main__":
    # Load all splits
    raw_train = load_split_lines(BASE_DIR, "train")
    raw_eval = load_split_lines(BASE_DIR, "eval")
    raw_test = load_split_lines(BASE_DIR, "test")

    train_sentences = parse_lines_to_sentences(raw_train)
    eval_sentences = parse_lines_to_sentences(raw_eval)
    test_sentences = parse_lines_to_sentences(raw_test)

    # Labels
    labels = extract_labels(train_sentences + eval_sentences + test_sentences)
    label2id = {l: i for i, l in enumerate(labels)}
    id2label = {i: l for l, i in label2id.items()}

    # Convert
    hf_train = convert_to_hf(train_sentences, label2id)
    hf_eval = convert_to_hf(eval_sentences, label2id)
    hf_test = convert_to_hf(test_sentences, label2id)
    train_ds = Dataset.from_list(hf_train)
    eval_ds = Dataset.from_list(hf_eval)
    test_ds = Dataset.from_list(hf_test)

    # Tokenizer
    tokenizer = AutoTokenizer.from_pretrained("airesearch/wangchanberta-base-att-spm-uncased")

    tokenized_train = train_ds.map(tokenize_and_align, batched=True, batch_size=1000, remove_columns=["tokens", "labels"])
    tokenized_eval = eval_ds.map(tokenize_and_align, batched=True, batch_size=1000, remove_columns=["tokens", "labels"])
    tokenized_test = test_ds.map(tokenize_and_align, batched=True, batch_size=1000, remove_columns=["tokens", "labels"])

    # Model
    model = AutoModelForTokenClassification.from_pretrained(
        "airesearch/wangchanberta-base-att-spm-uncased",
        num_labels=len(labels),
        id2label=id2label,
        label2id=label2id
    )

    # Training args
    training_args = TrainingArguments(
        output_dir="./NER_TrainingArgs_25",
        num_train_epochs=25,
        learning_rate=2e-5,
        per_device_train_batch_size=8,
        gradient_accumulation_steps=2,
        warmup_ratio=0.1,
        weight_decay=0.01,
        fp16=torch.cuda.is_available(),
        logging_strategy="steps",
        logging_steps=2,
        save_strategy="epoch",
        save_total_limit=1,
        eval_strategy="epoch",
        load_best_model_at_end=True,
        metric_for_best_model="loss",
        greater_is_better=False
    )

    # Trainer
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=tokenized_train,
        eval_dataset=tokenized_eval,
        tokenizer=tokenizer,
        compute_metrics=compute_metrics
    )

    # Train
    trainer.train()

    # Save model
    os.makedirs(SAVE_DIR, exist_ok=True)
    model.save_pretrained(SAVE_DIR)
    tokenizer.save_pretrained(SAVE_DIR)
    logger.info(f"บันทึกโมเดลที่ {SAVE_DIR}")

    # Plot losses
    history = trainer.state.log_history
    train_steps, train_losses, eval_steps, eval_losses = [], [], [], []

    for entry in history:
        if "loss" in entry:
            train_steps.append(entry["step"])
            train_losses.append(entry["loss"])
        if "eval_loss" in entry:
            eval_steps.append(entry["step"])
            eval_losses.append(entry["eval_loss"])

    plt.figure(figsize=(8, 5))
    plt.plot(train_steps, train_losses, label="Train Loss", marker="o")
    if eval_losses:
        plt.plot(eval_steps, eval_losses, label="Eval Loss", marker="x")
    plt.xlabel("Step")
    plt.ylabel("Loss")
    plt.title("Training vs Evaluation Loss Curve")
    plt.legend()
    plt.grid(True)
    plt.savefig(os.path.join(SAVE_DIR, "loss_curve.png"))
    plt.show()

    # Final test evaluation
    logger.info("ประเมินโมเดลบนชุด test ...")
    test_results = trainer.evaluate(tokenized_test)
    logger.info(f"ผลลัพธ์บน test set: {test_results}")
