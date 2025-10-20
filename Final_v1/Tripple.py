from transformers import AutoTokenizer, AutoModelForTokenClassification, pipeline

MODEL_PATH = r"./Final_v1/ner_modelfinal_25"
tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH)
model = AutoModelForTokenClassification.from_pretrained(MODEL_PATH)

nlp = pipeline(
    "token-classification",
    model=model,
    tokenizer=tokenizer,
    aggregation_strategy="simple"
    )

# extract POS NE
def run_ner_pos(text):
    outputs = nlp(text)
    tokens = []
    for out in outputs:
        word = out.get("word", "")
        if not word.strip():
            continue
        entity_group = out["entity_group"].split("|")
        pos = entity_group[0]
        ne = entity_group[1]
        tokens.append((word, pos, ne))
    return tokens

if __name__ == "__main__":
    text = "สมชายไปเที่ยวทะเล"
    triples = run_ner_pos(text)
    print("Extracted triples:")
    for t in triples:
        print(t)