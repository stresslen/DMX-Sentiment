"""Thí nghiệm: sửa lỗi chính tả bằng mô hình ViT5 (ProtonX legal-tc) cho cả 3 tập train/validation/test,
huấn luyện lại 7 mô hình trên dữ liệu đã sửa, rồi so sánh với kết quả trên dữ liệu gốc.

  A = dữ liệu gốc (outputs/)
  B = ViT5 sửa toàn bộ câu của train + validation + test, tiền xử lý và huấn luyện lại từ đầu (outputs/vit5/)

Chạy trong src/ (preprocess.py --vit5 gọi load_corrected() bên dưới, kết quả ViT5 được lưu cache):
  python preprocess.py --vit5 && python train_classical.py --vit5 && python train_phobert.py --vit5
  python analysis.py --vit5
  python vit5_correction.py          # so sánh A và B
"""
import json
import re
import time

import numpy as np
import pandas as pd
import torch
from sklearn.metrics import f1_score
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer

from analysis import predict_classical
from text_utils import OUT_ORIG, OUT_VIT5
from train_classical import LABEL_NAMES, MeanWord2Vec, compute_metrics  # noqa: F401  (MeanWord2Vec: để joblib nạp Word2Vec)

CORRECTOR = "protonx-models/protonx-legal-tc"  # T5 nền ViT5-base, giấy phép phi thương mại (v1.3-NC)
CHUNK_TOKENS = 120   # mô hình được huấn luyện với tối đa 128 token đầu vào
NUM_BEAMS = 10       # theo hướng dẫn trên model card
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
CACHE = OUT_ORIG / "processed"   # {split}_vit5.parquet: văn bản đã sửa của từng tập
N_BOOT = 2000
VARIANTS = {"A. Dữ liệu gốc": OUT_ORIG, "B. ViT5 sửa train + val + test": OUT_VIT5}


def split_chunks(text, tok):
    """Cắt câu dài thành đoạn <= CHUNK_TOKENS token, ưu tiên ranh giới câu."""
    sents = re.split(r"(?<=[.!?…\n])\s*", text.strip())
    chunks, cur = [], ""
    for s in filter(None, sents):
        cand = f"{cur} {s}".strip()
        if len(tok(cand).input_ids) <= CHUNK_TOKENS:
            cur = cand
            continue
        if cur:
            chunks.append(cur)
        words, cur = s.split(), ""
        for w in words:  # một câu quá dài -> cắt theo từ
            cand = f"{cur} {w}".strip()
            if cur and len(tok(cand).input_ids) > CHUNK_TOKENS:
                chunks.append(cur)
                cand = w
            cur = cand
    if cur:
        chunks.append(cur)
    return chunks or [text]


@torch.no_grad()
def correct_all(texts, batch_size=32):
    tok = AutoTokenizer.from_pretrained(CORRECTOR)
    model = AutoModelForSeq2SeqLM.from_pretrained(CORRECTOR).to(DEVICE).eval()
    pieces = [(i, c) for i, t in enumerate(texts) for c in split_chunks(t, tok)]
    order = sorted(range(len(pieces)), key=lambda k: len(pieces[k][1]))  # gom câu cùng độ dài cho nhanh
    fixed = [None] * len(pieces)
    for b in range(0, len(order), batch_size):
        idx = order[b:b + batch_size]
        enc = tok([pieces[k][1] for k in idx], return_tensors="pt", padding=True,
                  truncation=True, max_length=160).to(DEVICE)
        out = model.generate(**enc, num_beams=NUM_BEAMS, max_new_tokens=160, early_stopping=True)
        for k, y in zip(idx, tok.batch_decode(out, skip_special_tokens=True)):
            fixed[k] = y.strip()
        if b // batch_size % 200 == 0:
            print(f"  ViT5 {b}/{len(order)} đoạn", flush=True)
    joined = [[] for _ in texts]
    for (i, _), y in zip(pieces, fixed):
        joined[i].append(y)
    del model
    torch.cuda.empty_cache()
    return [" ".join(p) for p in joined], len(pieces)


def load_corrected(split, df):
    """Văn bản gốc của một tập đã được ViT5 sửa, đúng thứ tự dòng của df. ViT5 chạy lâu nên dùng lại cache nếu có."""
    cache = CACHE / f"{split}_vit5.parquet"
    if cache.exists():
        old = pd.read_parquet(cache, columns=["id", "text_vit5"])
        if len(old) == len(df) and (old["id"].to_numpy() == df["id"].to_numpy()).all():
            return old["text_vit5"].tolist()
    t0 = time.time()
    fixed, n_pieces = correct_all(df["text"].tolist())
    print(f"ViT5 {split}: {len(df)} câu, {n_pieces} đoạn, {time.time() - t0:.0f} giây", flush=True)
    CACHE.mkdir(parents=True, exist_ok=True)
    pd.DataFrame({"id": df["id"], "text_vit5": fixed}).to_parquet(cache, index=False)
    return fixed


def test_predictions(out):
    """Dự đoán trên tập test của 7 mô hình đã huấn luyện trong thư mục kết quả `out`."""
    test = pd.read_parquet(out / "processed" / "test.parquet")
    res = pd.read_csv(out / "metrics" / "classical_results.csv")
    preds, info = {}, {}
    for _, r in res.iterrows():
        slug = f"{r.model}_{r.feature}".replace(" ", "").replace("-", "").lower()
        preds[f"{r.model} + {r.feature}"] = predict_classical(slug, test["tokens_all"].tolist(), out / "models")
        info[f"{r.model} + {r.feature}"] = (r.val_f1_macro, r.imbalance_strategy)
    bert = pd.read_csv(out / "metrics" / "phobert_test_predictions.csv")
    assert (bert["id"].astype(str).to_numpy() == test["id"].astype(str).to_numpy()).all()
    preds["PhoBERT"] = bert["pred"].to_numpy()
    summary = json.loads((out / "metrics" / "phobert_results.json").read_text())["summary"]
    info["PhoBERT"] = (summary["val_f1_macro"], summary["imbalance_strategy"])
    return test, preds, info


def main():
    tests, preds, info = {}, {}, {}
    for v, out in VARIANTS.items():
        tests[v], preds[v], info[v] = test_predictions(out)
    (a, test_a), (b, test_b) = tests.items()
    assert (test_a["id"].to_numpy() == test_b["id"].to_numpy()).all()
    y = test_a["label"].to_numpy()

    rng = np.random.default_rng(42)
    idx = rng.integers(0, len(y), size=(N_BOOT, len(y)))
    rows = []
    for model in preds[a]:
        base = preds[a][model]
        base_boot = np.array([f1_score(y[i], base[i], average="macro") for i in idx])
        for v in VARIANTS:
            p = preds[v][model]
            m = compute_metrics(y, p)
            row = {"model": model, "variant": v, "imbalance_strategy": info[v][model][1],
                   "val_f1_macro": info[v][model][0], "accuracy": m["accuracy"], "f1_macro": m["f1_macro"],
                   "f1_neutral": f1_score(y, p, average=None)[1], "pred_changed": float((p != base).mean())}
            if v != a:
                diff = np.array([f1_score(y[i], p[i], average="macro") for i in idx]) - base_boot
                row.update(delta_f1=float(m["f1_macro"] - compute_metrics(y, base)["f1_macro"]),
                           delta_ci_low=float(np.percentile(diff, 2.5)),
                           delta_ci_high=float(np.percentile(diff, 97.5)))
            rows.append(row)
        print(f"{model:24s}", [round(r["f1_macro"], 4) for r in rows[-len(VARIANTS):]],
              f"delta CI [{rows[-1]['delta_ci_low']:+.4f}, {rows[-1]['delta_ci_high']:+.4f}]")
    pd.DataFrame(rows).to_csv(OUT_ORIG / "metrics" / "vit5_correction_results.csv", index=False)

    # Ví dụ: câu test mà PhoBERT đổi dự đoán giữa A và B
    pb_a, pb_b = preds[a]["PhoBERT"], preds[b]["PhoBERT"]
    test = test_b.assign(a=pb_a, b=pb_b)
    name = dict(enumerate(LABEL_NAMES))
    changed = test["text"] != test["text_orig"]
    stats = json.loads((OUT_VIT5 / "processed" / "preprocess_stats.json").read_text())
    examples = {
        "corrector": CORRECTOR, "num_beams": NUM_BEAMS,
        "correction_stats": {s: {k: v for k, v in stats[s].items() if k.startswith("vit5_")}
                             for s in ("train", "validation", "test")},
        "phobert_flips_fixed": int(((pb_a != y) & (pb_b == y)).sum()),
        "phobert_flips_broken": int(((pb_a == y) & (pb_b != y)).sum()),
        "flips": [{"text": r.text_orig, "vit5": r.text, "label": name[r.label], "before": name[r.a], "after": name[r.b]}
                  for r in test[pb_a != pb_b].itertuples()][:40],
        "corrections": test[changed].sample(40, random_state=1)
            .rename(columns={"text_orig": "text", "text": "text_vit5"})[["text", "text_vit5"]].to_dict("records"),
    }
    (OUT_ORIG / "metrics" / "vit5_correction_examples.json").write_text(
        json.dumps(examples, ensure_ascii=False, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()
