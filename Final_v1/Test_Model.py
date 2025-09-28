import torch
from transformers import AutoTokenizer, AutoModelForTokenClassification

MODEL_DIR = r"Final_v1\ner_modelfinal_v4"
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

tokenizer = AutoTokenizer.from_pretrained(MODEL_DIR)
tokenizer.model_max_length = 510
model = AutoModelForTokenClassification.from_pretrained(MODEL_DIR).to(DEVICE).eval()
id2label = {int(k): v for k, v in model.config.id2label.items()}

def tokenize(text):
    from attacut import tokenize
    return [t for t in tokenize(text) if t.strip()]

def split_label(label: str):
    parts = label.split()
    if len(parts) == 3:
        return parts  # POS, NER, CLS
    elif len(parts) == 2:
        return parts[0], parts[1], "O"
    elif len(parts) == 1:
        return parts[0], "O", "O"
    return "UNK", "O", "O"

def predict(text):
    tokens = tokenize(text)
    enc = tokenizer(tokens, is_split_into_words=True, return_tensors="pt", truncation=True, max_length=510)
    word_ids = enc.word_ids()
    inputs = {k: v.to(DEVICE) for k, v in enc.items()}
    logits = model(**inputs).logits[0]

    output, prev = [], None
    for i, wi in enumerate(word_ids):
        if wi is not None and wi != prev:
            label = id2label[int(logits[i].argmax())]
            pos, ner, cls = split_label(label)
            output.append((tokens[wi], pos, ner, cls))
        prev = wi
    return output

# Example usage
if __name__ == "__main__":
    text = "นายพลระดับสูงของกองทัพสหรัฐฯ หลายร้อยนายทั่วโลก ถูกเรียกไปร่วมประชุมร่วมกับรัฐมนตรีกลาโหมที่รัฐเวอร์จิเนียในสัปดาห์หน้า โดยไม่มีใครบอกได้ว่า จุดประสงค์ของการประชุมคืออะไร"
    for word, pos, ner, cls in predict(text):
        print(f"{word}\t{pos}")