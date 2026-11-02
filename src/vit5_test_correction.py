"""Thí nghiệm: sửa lỗi chính tả tập test bằng mô hình ViT5 (ProtonX legal-tc) rồi đánh giá lại 7 mô hình đã huấn luyện.

Không huấn luyện lại gì: chỉ thay văn bản đầu vào của tập test, phần tiền xử lý và mô hình giữ nguyên.
  A = văn bản gốc (kết quả hiện tại)
  B = ViT5 sửa toàn bộ câu
  C = ViT5 chỉ sửa câu bị nhận diện là "bẩn" (teencode, thiếu dấu, dính dấu câu, kéo dài ký tự)
"""
import json
import re
import time
from difflib import SequenceMatcher

import numpy as np
import pandas as pd
import torch
from sklearn.metrics import f1_score
from transformers import AutoModelForSeq2SeqLM, AutoModelForSequenceClassification, AutoTokenizer

from analysis import predict_classical
from preprocess import process_one
from text_utils import ROOT, TEENCODE, clean_tokens
from train_classical import LABEL_NAMES, MeanWord2Vec, compute_metrics  # noqa: F401  (MeanWord2Vec: để joblib nạp Word2Vec)

CORRECTOR = "protonx-models/protonx-legal-tc"  # T5 nền ViT5-base, giấy phép phi thương mại (v1.3-NC)
CHUNK_TOKENS = 120   # mô hình được huấn luyện với tối đa 128 token đầu vào
NUM_BEAMS = 10       # theo hướng dẫn trên model card
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
PROC = ROOT / "outputs" / "processed"
MET = ROOT / "outputs" / "metrics"
MOD = ROOT / "outputs" / "models"
N_BOOT = 2000

_VI_CHARS = re.compile("[àáảãạăằắẳẵặâầấẩẫậèéẻẽẹêềếểễệìíỉĩịòóỏõọôồốổỗộơờớởỡợùúủũụưừứửữựỳýỷỹỵđ]")
_TEEN = {k for k in TEENCODE if " " not in k}


def is_noisy(text):
    words = re.findall(r"\w+", text.lower())
    no_diacritics = len(words) >= 4 and sum(bool(_VI_CHARS.search(w)) for w in words) / len(words) < 0.15
    return (no_diacritics or any(w in _TEEN for w in words)
            or bool(re.search(r"(\w)\1{2,}", text.lower()))
            or bool(re.search(r"[a-zà-ỹđ][.,][A-Za-zÀ-ỹĐđ]", text)))


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
    joined = [[] for _ in texts]
    for (i, _), y in zip(pieces, fixed):
        joined[i].append(y)
    return [" ".join(p) for p in joined], len(pieces)


@torch.no_grad()
def predict_phobert(texts_seg, batch_size=128):
    path = MOD / "phobert" / "best"
    tok = AutoTokenizer.from_pretrained(path)
    model = AutoModelForSequenceClassification.from_pretrained(path).to(DEVICE).eval()
    preds = np.zeros(len(texts_seg), dtype=int)
    for b in range(0, len(texts_seg), batch_size):
        enc = tok(texts_seg[b:b + batch_size], return_tensors="pt", padding=True,
                  truncation=True, max_length=256).to(DEVICE)
        preds[b:b + batch_size] = model(**enc).logits.argmax(-1).cpu().numpy()
    return preds


def featurize(texts):
    norm, seg = zip(*map(process_one, texts))
    return list(seg), [" ".join(clean_tokens(s)) for s in seg]


def predict_all(seg, tokens):
    res = pd.read_csv(MET / "classical_results.csv")
    preds = {f"{r.model} + {r.feature}": predict_classical(
        f"{r.model}_{r.feature}".replace(" ", "").replace("-", "").lower(), tokens) for _, r in res.iterrows()}
    preds["PhoBERT"] = predict_phobert(seg)
    return preds


def main():
    test = pd.read_parquet(PROC / "test.parquet")
    y = test["label"].to_numpy()
    noisy = test["text"].apply(is_noisy).to_numpy()

    cache = PROC / "test_vit5.parquet"
    if cache.exists():  # ViT5 chạy ~5 phút; dùng lại kết quả đã sửa nếu có
        old = pd.read_parquet(cache)
        assert (old["id"].to_numpy() == test["id"].to_numpy()).all()
        fixed, n_pieces, correct_seconds = old["text_vit5"].tolist(), None, None
    else:
        t0 = time.time()
        fixed, n_pieces = correct_all(test["text"].tolist())
        correct_seconds = time.time() - t0
    test["text_vit5"] = fixed
    test["text_vit5_selective"] = np.where(noisy, test["text_vit5"], test["text"])
    test["noisy"] = noisy
    test.to_parquet(PROC / "test_vit5.parquet", index=False)

    changed = test["text_vit5"] != test["text"]
    sim = [SequenceMatcher(None, a, b).ratio() for a, b in zip(test["text"], test["text_vit5"])]
    corrector_stats = {
        "corrector": CORRECTOR, "num_beams": NUM_BEAMS, "pieces": n_pieces,
        "seconds": correct_seconds and round(correct_seconds, 1),
        "noisy_share": float(noisy.mean()),
        "changed_share_all": float(changed.mean()),
        "changed_share_noisy": float(changed[noisy].mean()),
        "changed_share_clean": float(changed[~noisy].mean()),   # sửa thừa trên câu sạch
        "mean_char_similarity": float(np.mean(sim)),
    }
    print(corrector_stats)

    variants = {"A. Gốc (hiện tại)": test["text"], "B. ViT5 toàn bộ": test["text_vit5"],
                "C. ViT5 chọn lọc": test["text_vit5_selective"]}
    preds = {v: predict_all(*featurize(t.tolist())) for v, t in variants.items()}

    rng = np.random.default_rng(42)
    idx = rng.integers(0, len(y), size=(N_BOOT, len(y)))
    rows = []
    for model in preds["A. Gốc (hiện tại)"]:
        base = preds["A. Gốc (hiện tại)"][model]
        base_boot = np.array([f1_score(y[i], base[i], average="macro") for i in idx])
        for v in variants:
            p = preds[v][model]
            m = compute_metrics(y, p)
            row = {"model": model, "variant": v, "accuracy": m["accuracy"], "f1_macro": m["f1_macro"],
                   "f1_neutral": f1_score(y, p, average=None)[1],
                   "f1_macro_noisy_subset": f1_score(y[noisy], p[noisy], average="macro"),
                   "pred_changed": float((p != base).mean())}
            if v != "A. Gốc (hiện tại)":
                diff = np.array([f1_score(y[i], p[i], average="macro") for i in idx]) - base_boot
                row.update(delta_f1=float(m["f1_macro"] - compute_metrics(y, base)["f1_macro"]),
                           delta_ci_low=float(np.percentile(diff, 2.5)),
                           delta_ci_high=float(np.percentile(diff, 97.5)))
            rows.append(row)
        print(model, [round(r["f1_macro"], 4) for r in rows[-3:]])
    out = pd.DataFrame(rows)
    out.to_csv(MET / "vit5_correction_results.csv", index=False)

    # Ví dụ: câu PhoBERT đổi dự đoán sau khi ViT5 sửa
    pb_a, pb_b = preds["A. Gốc (hiện tại)"]["PhoBERT"], preds["B. ViT5 toàn bộ"]["PhoBERT"]
    flips = test.assign(a=pb_a, b=pb_b)[pb_a != pb_b]
    name = dict(enumerate(LABEL_NAMES))
    examples = {
        "stats": corrector_stats,
        "phobert_flips_fixed": int(((pb_a != y) & (pb_b == y)).sum()),
        "phobert_flips_broken": int(((pb_a == y) & (pb_b != y)).sum()),
        "flips": [{"text": r.text, "vit5": r.text_vit5, "label": name[r.label], "before": name[r.a], "after": name[r.b]}
                  for r in flips.itertuples()][:40],
        "corrections_noisy": test[noisy & changed].sample(30, random_state=1)[["text", "text_vit5"]].to_dict("records"),
        "overcorrections_clean": test[~noisy & changed].sample(30, random_state=1)[["text", "text_vit5"]].to_dict("records"),
    }
    (MET / "vit5_correction_examples.json").write_text(json.dumps(examples, ensure_ascii=False, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()
