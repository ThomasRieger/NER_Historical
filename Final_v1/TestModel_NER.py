import re
from typing import List, Dict, Any
import torch
from transformers import AutoTokenizer, AutoModelForTokenClassification

MODEL_DIR = r"Final_v1\ner_modelfinal_v3"
MAX_LENGTH = 510
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
POSTPROCESS_RULES = True   # ตัดแท็กของเครื่องหมายวรรคตอนให้เป็น 'O' หลังทำนาย

def normalize_text(text: str) -> str:
    return (
        text.replace("\ufeff", "")   
            .replace("\u00A0", " ")  
            .replace("\u200B", "")   
    )

def safe_id2label(model) -> Dict[int, str]:
    if hasattr(model.config, "id2label"):
        raw = model.config.id2label
        if isinstance(raw, list):
            return {i: s for i, s in enumerate(raw)}
        if isinstance(raw, dict):
            return {int(k): v for k, v in raw.items()}
    if hasattr(model.config, "label2id"):
        inv = {v: k for k, v in model.config.label2id.items()}
        return {int(v): k for k, v in inv.items()}
    raise ValueError("ไม่พบ id2label/label2id ใน config ของโมเดล")

def is_punct_only(token: str) -> bool:
    return bool(token) and all(re.match(r"\W", ch) for ch in token)

def apply_simple_rules(tokens: List[str], tags: List[str]) -> List[str]:
    if not POSTPROCESS_RULES:
        return tags
    out = tags[:]
    for i, t in enumerate(tokens):
        if is_punct_only(t):
            out[i] = "O"
    return out

# attacut
def attacut_tokenize(text: str) -> List[str]:
    try:
        from attacut import tokenize as attacut_tok
    except Exception as e:
        raise ImportError("install attacut") from e
    toks = attacut_tok(text)
    return [t for t in toks if t and not t.isspace()]

# LOAD MODEL/TOKENIZER HF 
tokenizer = AutoTokenizer.from_pretrained(MODEL_DIR)
model = AutoModelForTokenClassification.from_pretrained(MODEL_DIR).to(DEVICE)
model.eval()
id2label = safe_id2label(model)

# INFERENCE 
def predict_from_tokens(tokens: List[str]) -> List[str]:
    if not tokens:
        return []
    enc = tokenizer(
        tokens,
        is_split_into_words=True,
        return_tensors="pt",
        truncation=True,
        padding=False,
        max_length=MAX_LENGTH,
    )
    word_ids = enc.word_ids(batch_index=0)  # mapping subword
    inputs = {k: v.to(DEVICE) for k, v in enc.items() if isinstance(v, torch.Tensor)}

    with torch.no_grad():
        logits = model(**inputs).logits[0]  # [seq_len, num_labels]

    preds, prev_wi = [], None
    for i, wi in enumerate(word_ids):
        if wi is None:
            continue
        if wi != prev_wi:  # เอาเฉพาะ subword แรกของคำ
            label_id = int(torch.argmax(logits[i]).cpu())
            preds.append(id2label[label_id])
        prev_wi = wi

    # ความยาวให้เท่าจำนวนคำ
    if len(preds) > len(tokens):
        preds = preds[:len(tokens)]
    elif len(preds) < len(tokens):
        preds += ["O"] * (len(tokens) - len(preds))

    return apply_simple_rules(tokens, preds)

def predict_from_text(text: str) -> List[Dict[str, str]]:
    text = normalize_text(text)
    tokens = attacut_tokenize(text)
    labels = predict_from_tokens(tokens)
    return [{"token": t, "label": l} for t, l in zip(tokens, labels)]

def predict_batch_texts(texts: List[str]) -> List[List[Dict[str, str]]]:
    token_lists: List[List[str]] = [attacut_tokenize(normalize_text(t)) for t in texts]
    if not token_lists:
        return []
    enc = tokenizer(
        token_lists,
        is_split_into_words=True,
        return_tensors="pt",
        truncation=True,
        padding=True,
        max_length=MAX_LENGTH,
    )
    inputs = {k: v.to(DEVICE) for k, v in enc.items() if isinstance(v, torch.Tensor)}
    with torch.no_grad():
        logits = model(**inputs).logits                 # [B, L, C]
        pred_ids = torch.argmax(logits, dim=-1).cpu().tolist()

    results: List[List[Dict[str, str]]] = []
    for bi, tokens in enumerate(token_lists):
        wi_list = enc.word_ids(batch_index=bi)
        seq_preds = pred_ids[bi]
        out_labels, prev_wi = [], None
        for i, wi in enumerate(wi_list):
            if wi is None:
                continue
            if wi != prev_wi:
                out_labels.append(id2label[int(seq_preds[i])])
            prev_wi = wi
        if len(out_labels) > len(tokens):
            out_labels = out_labels[:len(tokens)]
        elif len(out_labels) < len(tokens):
            out_labels += ["O"] * (len(tokens) - len(out_labels))
        out_labels = apply_simple_rules(tokens, out_labels)
        results.append([{"token": t, "label": l} for t, l in zip(tokens, out_labels)])
    return results

# tag span
def tags_to_spans(tokens: List[str], tags: List[str]) -> List[Dict[str, Any]]:
    spans = []
    cur_type, start = None, None

    def flush(end_idx: int):
        nonlocal cur_type, start
        if cur_type is not None and start is not None and end_idx > start:
            text = "".join(tokens[start:end_idx])
            spans.append({"type": cur_type, "text": text, "start": start, "end": end_idx})
        cur_type, start = None, None

    for i, tag in enumerate(tags):
        if not tag or tag == "O":
            flush(i); continue
        if tag.startswith("B_"):
            flush(i); cur_type = tag[2:]; start = i
        elif tag.startswith("I_"):
            if cur_type is None:
                cur_type = tag[2:]; start = i
        elif tag.startswith("E_"):
            ent_type = tag[2:]
            if cur_type is None:
                cur_type = ent_type; start = i
            flush(i + 1)
        else:
            flush(i); cur_type = tag; start = i
    flush(len(tags))
    return spans

# test
if __name__ == "__main__":
    sample = "นายอำเภอกะทู้ลงตรวจสอบการขุดดินบนเขากมลา อบต.กมลาแจ้งความเอาผิด ’ฝ่าฝืนคำสั่ง’"
    items = predict_from_text(sample)
    print("== Token-level ==")
    for it in items:
        print(f"{it['token']}\t{it['label']}")
    toks = [x["token"] for x in items]
    tags = [x["label"] for x in items]
    print("\n== Spans ==")
    for sp in tags_to_spans(toks, tags):
        print(sp)
