import os
import csv
from collections import Counter, defaultdict

# ====== ตั้งค่าโฟลเดอร์ ======
SPLITS = {
    "train": r"E:\Code\Project_Final\NER_Historical\Final_v1\AIFORTHAI-LST20Corpus\LST20_Corpus_final\train",
    "test":  r"E:\Code\Project_Final\NER_Historical\Final_v1\AIFORTHAI-LST20Corpus\LST20_Corpus_final\test",
    "eval":  r"E:\Code\Project_Final\NER_Historical\Final_v1\AIFORTHAI-LST20Corpus\LST20_Corpus_final\eval",
}

# ====== ฟังก์ชันนับ B_ ต่อโฟลเดอร์ ======
def count_b_tags_in_folder(folder_path):
    counter = Counter()
    for filename in os.listdir(folder_path):
        if not filename.lower().endswith(".txt"):
            continue
        filepath = os.path.join(folder_path, filename)
        # ใช้ errors="ignore" กันกรณีไฟล์ปน encoding แปลก ๆ
        with open(filepath, "r", encoding="utf-8", errors="ignore") as f:
            for raw in f:
                line = raw.strip()
                if not line:
                    continue
                parts = line.split("\t")
                if len(parts) >= 3:
                    ner_tag = parts[2].strip()
                    if ner_tag.startswith("B_"):
                        counter[ner_tag] += 1
    return counter

# ====== นับแยก split ======
split_counters = {split: count_b_tags_in_folder(path) for split, path in SPLITS.items()}

# รวมทั้งหมด
all_tags = set().union(*[set(c.keys()) for c in split_counters.values()])
total_counter = Counter()
for c in split_counters.values():
    total_counter.update(c)

# ====== แสดงผล ======
def pretty_print_split(split_name, counter):
    print(f"\nสรุปจำนวน class NER (เฉพาะ B_) : {split_name}")
    if not counter:
        print("  (ไม่พบ B_ ในชุดนี้)")
        return
    for tag in sorted(counter.keys()):
        print(f"{tag}: {counter[tag]:,}")

for split in ["train", "test", "eval"]:
    pretty_print_split(split, split_counters.get(split, Counter()))

print("\nสรุปรวมทุกชุด (train + test + eval)")
for tag in sorted(all_tags):
    print(f"{tag}: {total_counter[tag]:,}")

# ====== (ทางเลือก) บันทึกเป็น CSV ======
# ตั้ง True ถ้าต้องการไฟล์ CSV
SAVE_CSV = True
CSV_PATH = r"E:\Code\Project_Final\NER_Historical\Count_B_only_by_split.csv"

if SAVE_CSV:
    rows = []
    # แถวต่อ split
    for split in ["train", "test", "eval"]:
        c = split_counters.get(split, Counter())
        for tag in sorted(all_tags):
            rows.append([split, tag, c.get(tag, 0)])
    # แถวรวม
    for tag in sorted(all_tags):
        rows.append(["total", tag, total_counter.get(tag, 0)])

    os.makedirs(os.path.dirname(CSV_PATH), exist_ok=True)
    with open(CSV_PATH, "w", newline="", encoding="utf-8-sig") as f:
        writer = csv.writer(f)
        writer.writerow(["split", "label", "count"])
        writer.writerows(rows)
    print(f"\nบันทึกไฟล์ CSV แล้วที่: {CSV_PATH}")
