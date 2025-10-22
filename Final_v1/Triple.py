from transformers import AutoTokenizer, AutoModelForTokenClassification, pipeline

MODEL_PATH = r"./Final_v1/ner_modelfinal_v5"
# MODEL_PATH = r"./Final_v1/ner_modelfinal_25" 

try:
    tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH)
    model = AutoModelForTokenClassification.from_pretrained(MODEL_PATH)
except OSError:
    print(f"Error: Could not load model from {MODEL_PATH}")
    print("Path incorrect")
    exit()

nlp = pipeline(
    "token-classification",
    model=model,
    tokenizer=tokenizer,
    aggregation_strategy="simple"
)

def run_ner_pos(text):
    outputs = nlp(text)
    tokens = []
    # print(f"DEBUG: Raw model output:\n{outputs}\n")
    for out in outputs:
        word = out.get("word", "")
        if not word.strip():
            continue
        entity_label = out["entity_group"]
        entity_parts = entity_label.split("|")
        if len(entity_parts) >= 2:
            pos = entity_parts[0]
            ne = entity_parts[1]
        else:
            pos = "UNKNOWN" 
            ne = entity_label  
        tokens.append((word, pos, ne))
    return tokens

def find_all_spo(tokens):
    """
    Finds all Subject-Predicate-Object (SPO) triples.
    - Assumes the first Noun-chunk is the subject for all subsequent actions.
    - Chunks consecutive POS tags (NNs, VVs).
    - Stops an object-chunk when a new verb (VV) or conjunction (CC) is found.
    """
    triples = []
    i = 0
    subject = ""
    s_end_idx = -1
    while i < len(tokens):
        word, pos, ne = tokens[i]
        if pos.startswith("NN") or pos.startswith("PP"):
            subject += word
            s_end_idx = i
        elif subject: 
            break
        i += 1
    if not subject:
        return []

    while i < len(tokens):
        predicate, obj = "", ""
        p_end_idx = -1
        while i < len(tokens):
            word, pos, ne = tokens[i]
            if pos.startswith("VV"):
                predicate += word
                p_end_idx = i
            elif predicate: 
                break
            i += 1
        
        if not predicate:
            break 

        obj_i = p_end_idx + 1
        while obj_i < len(tokens):
            word, pos, ne = tokens[obj_i]
            if pos.startswith("VV") or pos.startswith("CC"):
                break
            if pos.startswith("NN") or pos.startswith("NU") or pos.startswith("CL"):
                obj += word
            obj_i += 1
        if obj:
            triples.append((subject, predicate, obj))
        i = obj_i
    return triples

if __name__ == "__main__":
    text = "เป็นซากเรือรบของญี่ปุ่นสมัยสงครามโลกครั้งที่ 2 ซึ่งในระหว่างสงครามกองทัพญี่ปุ่นได้ใช้พื้นที่ในตำบลปากจั่น อำเภอกระบุรี เป็นท่าเรือเพื่อส่งกำลังบำรุงไปยังประเทศพม่า ซากเรืออยู่บริเวณสะพานข้ามแม่น้ำละอุ่นจะสังเกตเห็นได้ในเวลาน้ำลง"
    # Model Extract
    pos_triples = run_ner_pos(text)
    
    print("Extracted POS/NE triples:")
    for t in pos_triples:
        print(t)

    # find patterm
    spo_triples = find_all_spo(pos_triples)
    
    print("\nExtracted SPO triple(s):")
    if spo_triples:
        for s, p, o in spo_triples:
            print(f"{s} -> {p} -> {o}")
    else:
        print("Could not find any Subject-Predicate-Object patterns.")