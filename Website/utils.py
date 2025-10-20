import os, re, torch, pandas as pd, streamlit as st, requests, base64, io
from typing import List, Dict, Any
from transformers import AutoTokenizer, AutoModelForTokenClassification

# --- Config ---
MODEL_DIR = os.path.join("Final_v1", "ner_modelfinal_25")
GITHUB_REPO = "ThomasRieger/NER_Historical_Storage"
GITHUB_BRANCH = "main"
GITHUB_FILE_PATH = "Saved_Data.csv"
GITHUB_TOKEN = st.secrets.get("GITHUB_TOKEN", "")
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
MAX_LENGTH = 510

# --- Load Model ---
@st.cache_resource
def load_ner_model():
    tokenizer = AutoTokenizer.from_pretrained(MODEL_DIR, use_fast=True)
    model = AutoModelForTokenClassification.from_pretrained(MODEL_DIR).to(DEVICE)
    model.eval()
    id2label = {int(k): v for k, v in model.config.id2label.items()}
    return tokenizer, model, id2label

tokenizer, model, id2label = load_ner_model()

# --- Text Utilities ---
def normalize_text(text: str) -> str:
    return text.replace("\ufeff", "").replace("\u00A0", " ").replace("\u200B", "")

def is_punct_only(token: str) -> bool:
    return bool(token) and all(re.match(r"\W", ch) for ch in token)

def apply_simple_rules(tokens: List[str], tags: List[str]) -> List[str]:
    return [t if not is_punct_only(tok) else "O" for tok, t in zip(tokens, tags)]

def attacut_tokenize(text: str) -> List[str]:
    from attacut import tokenize as attacut_tok
    return [t for t in attacut_tok(text) if t and not t.isspace()]

# --- Prediction ---
def predict_from_tokens(tokens: List[str]) -> List[str]:
    enc = tokenizer(tokens, is_split_into_words=True, return_tensors="pt", truncation=True, max_length=MAX_LENGTH)
    inputs = {k: v.to(DEVICE) for k, v in enc.items() if isinstance(v, torch.Tensor)}
    word_ids = enc.word_ids(batch_index=0)
    with torch.no_grad(): logits = model(**inputs).logits[0]
    preds, prev_wi = [], None
    for i, wi in enumerate(word_ids):
        if wi is None: continue
        if wi != prev_wi:
            label_id = int(torch.argmax(logits[i]).cpu())
            preds.append(id2label[label_id])
        prev_wi = wi
    return apply_simple_rules(tokens, preds[:len(tokens)])

def predict_from_text(text: str) -> List[Dict[str, str]]:
    text = normalize_text(text)
    tokens = attacut_tokenize(text)
    labels = predict_from_tokens(tokens)
    return [{"token": t, "label": l} for t, l in zip(tokens, labels)]

# --- Extraction ---
def extract_named_entities(text: str) -> List[tuple]:
    return [(ent["token"], ent["label"]) for ent in predict_from_text(text)]

def extract_triples(text: str) -> List[tuple]:
    ents = predict_from_text(text)
    return [(ents[i]["token"], "related_to", ents[i + 1]["token"]) for i in range(0, len(ents) - 1, 2)]

# --- GitHub Push ---
def save_to_database(entry: Dict[str, str]):
    api_url = f"https://api.github.com/repos/{GITHUB_REPO}/contents/{GITHUB_FILE_PATH}"
    headers = {"Authorization": f"token {GITHUB_TOKEN}"}
    new_df = pd.DataFrame([entry])

    try:
        response = requests.get(api_url, headers=headers, params={"ref": GITHUB_BRANCH})
        if response.status_code == 200:
            content = base64.b64decode(response.json()["content"]).decode("utf-8")
            sha = response.json()["sha"]
            existing_df = pd.read_csv(io.StringIO(content))
            combined_df = pd.concat([existing_df, new_df], ignore_index=True).drop_duplicates(subset=["text"])
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
    except Exception as e:
        st.error(f"Exception during GitHub save: {e}")

# --- GitHub Pull ---
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
            return pd.DataFrame()
        else:
            st.error(f"GitHub GET failed: {response.status_code}")
            return pd.DataFrame()
    except Exception as e:
        st.error(f"Exception during GitHub pull: {e}")
        return pd.DataFrame()