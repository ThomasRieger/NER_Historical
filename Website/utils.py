import os
import pandas as pd

DATABASE_PATH = r"Website\user_database.csv"

def extract_named_entities(text):
    return []

def extract_triples(text):
    return []

def save_to_database(entry):
    new_entry_df = pd.DataFrame([entry])

    if os.path.exists(DATABASE_PATH):
        existing_df = pd.read_csv(DATABASE_PATH)

        combined_df = pd.concat([existing_df, new_entry_df], ignore_index=True)
        combined_df.drop_duplicates(subset=["text"], inplace=True)

        combined_df.to_csv(DATABASE_PATH, index=False, encoding='utf-8-sig')

    else:
        new_entry_df.to_csv(DATABASE_PATH, index=False, encoding='utf-8-sig')
