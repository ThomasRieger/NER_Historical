from transformers import AutoTokenizer, AutoModelForTokenClassification, pipeline

MODEL_PATH = r"./Final_v1/ner_modelfinal_v5"

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
    print(tokens)
    return tokens

def extract_triples(tokens):
    # entity positions
    entities = []
    current_entity = []
    current_type = None
    start_idx = None

    for idx, (word, pos, ne) in enumerate(tokens):
        if ne.startswith("B_"):
            if current_entity:
                entities.append(("".join(current_entity), current_type, start_idx))
            current_entity = [word]
            current_type = ne[2:]
            start_idx = idx
        elif ne.startswith("I_") and current_entity:
            current_entity.append(word)
        else:
            if current_entity:
                entities.append(("".join(current_entity), current_type, start_idx))
                current_entity, current_type, start_idx = [], None, None
    if current_entity:
        entities.append(("".join(current_entity), current_type, start_idx))

    # find verbs
    verbs = [(word, idx) for idx, (word, pos, ne) in enumerate(tokens) if pos.startswith("V")]

    # extract triples
    triples = []
    for verb, v_idx in verbs:
        before = [e for e in entities if e[2] < v_idx]
        after = [e for e in entities if e[2] > v_idx]

        if before and after:
            subject = before[-1][0]  # before
            obj = after[0][0]        # after
            triples.append((subject, verb, obj))

    return triples

# extract triple
def extract_triples_from_text(text):
    tokens = run_ner_pos(text)
    triples = extract_triples(tokens)
    return triples

if __name__ == "__main__":
    text = "สมชายไปเที่ยวทะเล"
    triples = extract_triples_from_text(text)
    print("Extracted triples:")
    for t in triples:
        print(t)
