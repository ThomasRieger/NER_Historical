import os, glob, logging, torch
from typing import List, Dict
from datasets import Dataset
from transformers import (
    AutoTokenizer, AutoModelForTokenClassification,
    TrainingArguments, Trainer
)

BASE_DIR = r"Final_v1\AIFORTHAI-LST20Corpus\LST20_Corpus_final"
SPLIT = "train"
ENCODING = "utf-8"
SAVE_DIR = "./ner_modelfinal_v3"

logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
logger = logging.getLogger(__name__)

# load txt files
def load_split_lines(base_dir: str, split: str = "train") -> List[str]:
    pattern = os.path.join(base_dir, split, "*.txt")
    files = sorted(glob.glob(pattern))
    if not files:
        raise FileNotFoundError(f"ไม่พบไฟล์ที่ {pattern}")
    logger.info(f"{split}: พบ {len(files)} ไฟล์")
    lines = []
    for fp in files:
        with open(fp, encoding=ENCODING) as f:
            lines.extend(f.readlines())
        lines.append("\n")
    return lines

raw_lines = load_split_lines(BASE_DIR, SPLIT)

# sentence parse
def parse_lines_to_sentences(raw: List[str]) -> List[List[List[str]]]:
    sentences, current = [], []
    for i, line in enumerate(raw):
        line = line.rstrip("\n")
        if not line:
            if current:
                sentences.append(current)
                current = []
        else:
            parts = line.split("\t")
            if len(parts) == 4:
                current.append(parts)
            else:
                logger.warning(f"ข้ามบรรทัดผิดรูปแบบ ({i}): {line}")
    if current:
        sentences.append(current)
    logger.info(f"รวม {len(sentences)} ประโยค (ยาวสุด {max(map(len, sentences))} token)")
    return sentences

train_sentences = parse_lines_to_sentences(raw_lines)

# label
def extract_labels(sentences: List[List[List[str]]]) -> List[str]:
    labels = sorted({f"{tok[1]}|{tok[2]}|{tok[3]}" for sent in sentences for tok in sent})
    logger.info(f"พบ label รวม {len(labels)} ชนิด: {labels}")
    return labels

labels = extract_labels(train_sentences)
label2id = {l: i for i, l in enumerate(labels)}
id2label = {i: l for l, i in label2id.items()}

# huggin format
def convert_to_hf(sentences: List[List[List[str]]]) -> List[Dict]:
    data = []
    for sent in sentences:
        tokens = [w[0] for w in sent]
        tags = [label2id[f"{w[1]}|{w[2]}|{w[3]}"] for w in sent]
        data.append({"tokens": tokens, "labels": tags})
    return data

hf_train = convert_to_hf(train_sentences)
train_ds = Dataset.from_list(hf_train)

# tokenizer
tokenizer = AutoTokenizer.from_pretrained(
    "airesearch/wangchanberta-base-att-spm-uncased"
)

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

tokenized_train = train_ds.map(
    tokenize_and_align,
    batched=True,
    batch_size=1000,
    remove_columns=["tokens", "labels"]
)

# load model
model = AutoModelForTokenClassification.from_pretrained(
    "airesearch/wangchanberta-base-att-spm-uncased",
    num_labels=len(labels),
    id2label=id2label,
    label2id=label2id
)

training_args = TrainingArguments(
    output_dir="./NER_TrainingArgs",
    num_train_epochs=100,
    learning_rate=5e-5,
    per_device_train_batch_size=8,
    gradient_accumulation_steps=2,
    fp16=torch.cuda.is_available(),
    logging_strategy="steps",
    logging_steps=50,
    save_strategy="steps",
    save_steps=500,
    save_total_limit=1,
    eval_strategy="no"
)

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=tokenized_train,
    tokenizer=tokenizer
)

trainer.train()

os.makedirs(SAVE_DIR, exist_ok=True)
model.save_pretrained(SAVE_DIR)
tokenizer.save_pretrained(SAVE_DIR)
logger.info(f"✅ บันทึกโมเดลที่ {SAVE_DIR}")