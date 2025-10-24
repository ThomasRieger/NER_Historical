import os, re, torch, pandas as pd, streamlit as st, requests, base64, io
from typing import List, Dict, Any
from transformers import AutoTokenizer, AutoModelForTokenClassification

# config
MODEL_DIR = os.path.join("Final_v1", "ner_modelfinal_25")
GITHUB_REPO = "ThomasRieger/NER_Historical_Storage"
GITHUB_BRANCH = "main"
GITHUB_FILE_PATH = "Saved_Data.csv"
GITHUB_TOKEN = st.secrets.get("GITHUB_TOKEN", "")
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
MAX_LENGTH = 510

# model
@st.cache_resource
def load_ner_model():
    try:
        tokenizer = AutoTokenizer.from_pretrained(MODEL_DIR, use_fast=True)
        model = AutoModelForTokenClassification.from_pretrained(MODEL_DIR).to(DEVICE)
        model.eval()
        id2label = {int(k): v for k, v in model.config.id2label.items()}
    except OSError:
        st.error(f"Error: Could not load model from {MODEL_DIR}. Path incorrect.")
        return None, None, None
    return tokenizer, model, id2label

# Load model components
tokenizer, model, id2label = load_ner_model()

def normalize_text(text: str) -> str:
    return text.replace("\ufeff", "").replace("\u00A0", " ").replace("\u200B", "")

def is_punct_only(token: str) -> bool:
    return bool(token) and all(re.match(r"\W", ch) for ch in token)

def apply_simple_rules(tokens: List[str], tags: List[str]) -> List[str]:
    return [t if not is_punct_only(tok) else "O" for tok, t in zip(tokens, tags)]

def attacut_tokenize(text: str) -> List[str]:
    try:
        from attacut import tokenize as attacut_tok
    except ImportError:
        st.error("Attacut library not found. Please install it: pip install attacut")
        return text.split() 
    return [t for t in attacut_tok(text) if t and not t.isspace()]

# model prediction
def predict_from_tokens(tokens: List[str]) -> List[str]:
    if tokenizer is None or model is None or id2label is None:
        return ["O"] * len(tokens)
        
    enc = tokenizer(tokens, is_split_into_words=True, return_tensors="pt", truncation=True, max_length=MAX_LENGTH)
    inputs = {k: v.to(DEVICE) for k, v in enc.items() if isinstance(v, torch.Tensor)}
    word_ids = enc.word_ids(batch_index=0)
    
    with torch.no_grad(): 
        logits = model(**inputs).logits[0]
        
    preds, prev_wi = [], None
    for i, wi in enumerate(word_ids):
        if wi is None: continue
        if wi != prev_wi:
            label_id = int(torch.argmax(logits[i]).cpu())
            preds.append(id2label[label_id])
        prev_wi = wi
        
    if len(preds) < len(tokens):
         preds.extend(["O"] * (len(tokens) - len(preds)))
         
    return apply_simple_rules(tokens, preds[:len(tokens)])

def extract_named_entities(text: str) -> List[tuple]:
    if model is None: 
        return []
        
    text = normalize_text(text)
    tokens = attacut_tokenize(text)
    labels = predict_from_tokens(tokens)
    
    return list(zip(tokens, labels))

def _find_all_spo(tokens: List[tuple]) -> List[tuple]:
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

    i = s_end_idx + 1
    
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

def extract_triples(text: str) -> List[tuple]:
    if model is None: 
        return []

    text = normalize_text(text)
    tokens = attacut_tokenize(text)
    labels = predict_from_tokens(tokens)

    pos_ne_tokens = []
    for token, complex_label in zip(tokens, labels):
        if not token.strip():
            continue
        entity_parts = complex_label.split("|")
        if len(entity_parts) >= 2:
            pos = entity_parts[0]
            ne = entity_parts[1]
        else:
            pos = "UNKNOWN" 
            ne = complex_label
        pos_ne_tokens.append((token, pos, ne))
    
    return _find_all_spo(pos_ne_tokens)


# save data to github
def save_to_database(entry: Dict[str, str]):
    api_url = f"https://api.github.com/repos/{GITHUB_REPO}/contents/{GITHUB_FILE_PATH}"
    headers = {"Authorization": f"token {GITHUB_TOKEN}"}

    db_entry = {k: str(v) for k, v in entry.items()}
    new_df = pd.DataFrame([db_entry])

    try:
        response = requests.get(api_url, headers=headers, params={"ref": GITHUB_BRANCH})
        if response.status_code == 200:
            content = base64.b64decode(response.json()["content"]).decode("utf-8")
            sha = response.json()["sha"]
            existing_df = pd.read_csv(io.StringIO(content))
            combined_df = pd.concat([existing_df, new_df], ignore_index=True).drop_duplicates(subset=["text"], keep='last')
        elif response.status_code == 404:
            combined_df = new_df
            sha = None
        else:
            st.error(f"GitHub GET failed: {response.status_code}")
            return

        csv_content = combined_df.to_csv(index=False, encoding="utf-8-sig")
        encoded = base64.b64encode(csv_content.encode()).decode()
        payload = {
            "message": "Update user_database.csv",
            "content": encoded,
            "branch": GITHUB_BRANCH
        }
        if sha:
            payload["sha"] = sha

        put_response = requests.put(api_url, headers=headers, json=payload)
        if put_response.status_code not in [200, 201]:
             st.error(f"GitHub PUT failed: {put_response.status_code} - {put_response.text}")

    except Exception as e:
        st.error(f"Exception during GitHub save: {e}")

# pull data github
def pull_from_github() -> pd.DataFrame:
    api_url = f"https://api.github.com/repos/{GITHUB_REPO}/contents/{GITHUB_FILE_PATH}"
    headers = {"Authorization": f"token {GITHUB_TOKEN}"}

    try:
        response = requests.get(api_url, headers=headers, params={"ref": GITHUB_BRANCH})
        if response.status_code == 200:
            content = base64.b64decode(response.json()["content"]).decode("utf-8")
            df = pd.read_csv(io.StringIO(content))
            st.success("✅ Pulled latest data from GitHub.")
            return df
        elif response.status_code == 404:
            st.warning("⚠️ File not found on GitHub.")
            return pd.DataFrame(columns=["text", "clean_ner", "triples", "raw_ner"])
        else:
            st.error(f"GitHub GET failed: {response.status_code}")
            return pd.DataFrame(columns=["text", "clean_ner", "triples", "raw_ner"])
    except Exception as e:
        st.error(f"Exception during GitHub pull: {e}")
        return pd.DataFrame(columns=["text", "clean_ner", "triples", "raw_ner"])

