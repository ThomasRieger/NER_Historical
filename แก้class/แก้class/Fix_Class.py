import os
from collections import Counter

# ====== แหล่งข้อมูล (ต้นฉบับ) ======
SPLITS_SRC = {
    "train": r"E:\Code\Project_Final\NER_Historical\Final_v1\AIFORTHAI-LST20Corpus\LST20_Corpus_final\train",
    "test":  r"E:\Code\Project_Final\NER_Historical\Final_v1\AIFORTHAI-LST20Corpus\LST20_Corpus_final\test",
    "eval":  r"E:\Code\Project_Final\NER_Historical\Final_v1\AIFORTHAI-LST20Corpus\LST20_Corpus_final\eval",
}

# ====== ปลายทาง (ไฟล์ที่แก้ไขแล้ว) ======
BASE_OUT = r"E:\Code\Project_Final\NER_Historical\Final_v1\Fix_Class"

# ====== Mapping: เฉพาะแท็กที่ระบุ ======
REMAP = {
    "B_BOLE": "B_ROLE", "I_BOLE": "I_ROLE", "E_BOLE": "E_ROLE",
    "B_BRN": "O", "I_BRN": "O", "E_BRN": "O",
    "B_CIT": "B_PRO", "I_CIT": "I_PRO", "E_CIT": "E_PRO",
    "B_DATE": "B_DTM", "I_DATE": "I_DTM", "E_DATE": "E_DTM",
    "B_EVO": "O", "I_EVO": "O", "E_EVO": "O",
    "B_LSTY": "B_STY", "I_LSTY": "I_STY", "E_LSTY": "E_STY",
    "B_PRL": "O", "I_PRL": "O", "E_PRL": "O",
    "B_ROG": "B_ORG", "I_ROG": "I_ORG", "E_ROG": "E_ORG",
    "B_TRM": "O", "I_TRM": "O", "E_TRM": "O",
    "B_": "O",  # เคสแท็กเพี้ยนที่เป็น "B_" เฉย ๆ
}

# ====== ตัวนับสถิติ ======
replace_counts_by_split = {k: Counter() for k in SPLITS_SRC.keys()}
total_replace_counts = Counter()

def ensure_dir(path):
    os.makedirs(path, exist_ok=True)

def process_file(src_path, dst_path, split_key):
    out_lines = []
    changed = False

    with open(src_path, "r", encoding="utf-8", errors="ignore") as f:
        for raw in f:
            line = raw.rstrip("\n")
            if not line.strip():
                out_lines.append(line)
                continue

            parts = line.split("\t")
            if len(parts) >= 3:
                old_tag = parts[2].strip()
                new_tag = REMAP.get(old_tag, old_tag)
                if new_tag != old_tag:
                    replace_counts_by_split[split_key][f"{old_tag}->{new_tag}"] += 1
                    total_replace_counts[f"{old_tag}->{new_tag}"] += 1
                    parts[2] = new_tag
                    changed = True
                out_lines.append("\t".join(parts))
            else:
                out_lines.append(line)

    # เขียนผลลัพธ์ไปยังไฟล์ปลายทาง (เสมอ เพื่อให้ได้สำเนาครบทุกไฟล์)
    with open(dst_path, "w", encoding="utf-8", errors="ignore") as fw:
        fw.write("\n".join(out_lines) + "\n")

    return changed

def process_split(split_key, src_folder, dst_folder):
    ensure_dir(dst_folder)
    total_files = 0
    changed_files = 0

    for filename in os.listdir(src_folder):
        if not filename.lower().endswith(".txt"):
            continue
        total_files += 1
        src_path = os.path.join(src_folder, filename)
        dst_path = os.path.join(dst_folder, filename)
        changed = process_file(src_path, dst_path, split_key)
        if changed:
            changed_files += 1

    print(f"- {split_key}: เขียนไฟล์ {changed_files}/{total_files} ไฟล์ที่มีการเปลี่ยนแท็ก (รวมทั้งหมดถูกคัดลอกไปยังปลายทางแล้ว)")

# ====== Run ======
print(f"ปลายทาง: {BASE_OUT}")
for split_key, src_folder in SPLITS_SRC.items():
    dst_folder = os.path.join(BASE_OUT, split_key)
    print(f"ประมวลผลชุด {split_key}")
    process_split(split_key, src_folder, dst_folder)

# ====== รายงานผล ======
print("\nสรุปการแทนที่ (แยกตาม split):")
for split_key in ["train", "test", "eval"]:
    cnt = replace_counts_by_split[split_key]
    if not cnt:
        print(f"- {split_key}: (ไม่มีการแทนที่)")
        continue
    print(f"- {split_key}:")
    for k, v in sorted(cnt.items(), key=lambda x: (-x[1], x[0])):
        print(f"  {k}: {v:,}")

print("\nสรุปรวมทุกชุด:")
if not total_replace_counts:
    print("(ไม่มีการแทนที่ใด ๆ)")
else:
    for k, v in sorted(total_replace_counts.items(), key=lambda x: (-x[1], x[0])):
        print(f"{k}: {v:,}")

print(f"\n✅ เสร็จสิ้น! ไฟล์ผลลัพธ์อยู่ที่: {BASE_OUT}\\train, {BASE_OUT}\\test, {BASE_OUT}\\eval")
