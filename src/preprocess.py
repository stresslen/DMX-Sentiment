"""Bước 2 - Tiền xử lý: chuẩn hóa, tách từ (pyvi), loại stopword. Lưu kết quả ra outputs/processed/."""
import json
from multiprocessing import Pool

import pandas as pd

from text_utils import ROOT, clean_tokens, load_stopwords, normalize, segment

DATA_DIR = ROOT / "data" / "DMX-Sentiment-v1" / "parquet"
OUT_DIR = ROOT / "outputs" / "processed"
SPLITS = ("train", "validation", "test")


def process_one(text):
    norm = normalize(text)
    seg = segment(norm)
    return norm, seg


def main():
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    stopwords = load_stopwords()
    stats = {}
    for split in SPLITS:
        df = pd.read_parquet(DATA_DIR / f"{split}.parquet")[["id", "text", "label", "label_name", "category"]]
        with Pool() as pool:
            norm, seg = zip(*pool.map(process_one, df["text"], chunksize=500))
        df["text_norm"] = norm                      # đã chuẩn hóa, chưa tách từ
        df["text_seg"] = seg                        # đã tách từ, còn dấu câu (đầu vào PhoBERT)
        df["tokens_all"] = [" ".join(clean_tokens(s)) for s in seg]               # bỏ dấu câu, giữ stopword
        df["tokens"] = [" ".join(clean_tokens(s, stopwords)) for s in seg]       # bỏ dấu câu + stopword
        df.to_parquet(OUT_DIR / f"{split}.parquet", index=False)

        n_raw = df["text"].str.split().str.len()
        n_all = df["tokens_all"].str.split().str.len()
        n_tok = df["tokens"].str.split().str.len()
        stats[split] = {
            "rows": len(df),
            "avg_syllables_raw": round(n_raw.mean(), 2),
            "avg_words_segmented": round(n_all.mean(), 2),
            "avg_words_no_stopword": round(n_tok.mean(), 2),
            "empty_after_stopword": int((n_tok == 0).sum()),
        }
        print(split, stats[split])

    vocab_all = {w for s in pd.read_parquet(OUT_DIR / "train.parquet")["tokens_all"] for w in s.split()}
    vocab_tok = {w for s in pd.read_parquet(OUT_DIR / "train.parquet")["tokens"] for w in s.split()}
    stats["train_vocab_with_stopword"] = len(vocab_all)
    stats["train_vocab_no_stopword"] = len(vocab_tok)
    stats["n_stopwords"] = len(stopwords)
    (OUT_DIR / "preprocess_stats.json").write_text(json.dumps(stats, ensure_ascii=False, indent=2), encoding="utf-8")

    sample = pd.read_parquet(OUT_DIR / "train.parquet").sample(8, random_state=7)
    sample[["text", "text_norm", "text_seg", "tokens"]].to_csv(OUT_DIR / "examples.csv", index=False, encoding="utf-8-sig")


if __name__ == "__main__":
    main()
