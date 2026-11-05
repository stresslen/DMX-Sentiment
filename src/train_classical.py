"""Bước 3-5: trích chọn đặc trưng (BoW, TF-IDF, Word2Vec) + Naive Bayes / SVM, đánh giá trên tập test.

Xử lý mất cân bằng: mỗi tổ hợp thử 5 chiến lược (không xử lý, trọng số lớp, under/over/hybrid sampling).
Chiến lược và siêu tham số được chọn theo Macro-F1 trên tập validation; tập test chỉ dùng cho kết quả cuối.
"""
import json
import time

import joblib
from joblib import Parallel, delayed
import numpy as np
import pandas as pd
from gensim.models import Word2Vec
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.metrics import (accuracy_score, classification_report, confusion_matrix,
                             f1_score, precision_recall_fscore_support, recall_score)
from sklearn.naive_bayes import GaussianNB, MultinomialNB
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC, LinearSVC

from imbalance import class_counts, describe, resample_index, sampling_grid
from plot_style import grouped_bars, plot_confusion
from text_utils import OUT

SEED = 42
PROC = OUT / "processed"
MET = OUT / "metrics"
FIG = OUT / "figures"
MOD = OUT / "models"
LABEL_NAMES = ["dissatisfied", "neutral", "satisfied"]
N_JOBS = 8  # số tiến trình song song khi thử lưới (SVM-RBF chậm)


def load():
    return {s: pd.read_parquet(PROC / f"{s}.parquet") for s in ("train", "validation", "test")}


def compute_metrics(y_true, y_pred):
    p, r, f, _ = precision_recall_fscore_support(y_true, y_pred, average="macro", zero_division=0)
    pw, rw, fw, _ = precision_recall_fscore_support(y_true, y_pred, average="weighted", zero_division=0)
    return {
        "accuracy": accuracy_score(y_true, y_pred),
        "precision_macro": p, "recall_macro": r, "f1_macro": f,
        "precision_weighted": pw, "recall_weighted": rw, "f1_weighted": fw,
    }


# ---------------------------------------------------------------- Đặc trưng
class MeanWord2Vec:
    """Huấn luyện Word2Vec (skip-gram) trên tập train, biểu diễn câu = trung bình vector từ."""

    def __init__(self, dim=200, window=5, min_count=2, epochs=15):
        self.params = dict(vector_size=dim, window=window, min_count=min_count, epochs=epochs,
                           sg=1, negative=10, workers=8, seed=SEED)

    def fit(self, docs):
        self.model = Word2Vec(sentences=[d.split() for d in docs], **self.params)
        self.dim = self.model.vector_size
        return self

    def transform(self, docs):
        wv = self.model.wv
        out = np.zeros((len(docs), self.dim), dtype=np.float32)
        for i, d in enumerate(docs):
            vecs = [wv[w] for w in d.split() if w in wv]
            if vecs:
                out[i] = np.mean(vecs, axis=0)
        return out


def build_features(data, text_col="tokens_all"):
    tr, va, te = (data[s][text_col].tolist() for s in ("train", "validation", "test"))
    feats = {}

    bow = CountVectorizer(ngram_range=(1, 2), min_df=2, token_pattern=r"\S+")
    feats["BoW"] = (bow, bow.fit_transform(tr), bow.transform(va), bow.transform(te))

    tfidf = TfidfVectorizer(ngram_range=(1, 2), min_df=2, sublinear_tf=True, token_pattern=r"\S+")
    feats["TF-IDF"] = (tfidf, tfidf.fit_transform(tr), tfidf.transform(va), tfidf.transform(te))

    w2v = MeanWord2Vec().fit(tr)
    scaler = StandardScaler().fit(w2v.transform(tr))
    Xtr, Xva, Xte = (scaler.transform(w2v.transform(x)) for x in (tr, va, te))
    feats["Word2Vec"] = ((w2v, scaler), Xtr, Xva, Xte)
    return feats


# ---------------------------------------------------------------- Mô hình + lưới siêu tham số + xử lý mất cân bằng
STRATEGIES = ["Không xử lý", "Trọng số lớp", "Undersampling", "Oversampling", "Hybrid sampling"]


def model_grid(model_name, feat_name, y_tr):
    """Danh sách (chiến lược mất cân bằng, số mẫu mục tiêu hoặc None, siêu tham số, hàm tạo mô hình).

    - Không xử lý: huấn luyện trên dữ liệu gốc, mô hình mặc định.
    - Trọng số lớp: NB bỏ tiên nghiệm lớp (prior đều); SVM dùng class_weight="balanced".
    - Under/Over/Hybrid sampling: mô hình mặc định trên tập train đã lấy mẫu lại (xem imbalance.py).
    """
    dense = feat_name == "Word2Vec"
    if model_name == "Naive Bayes" and dense:  # đặc trưng liên tục (có giá trị âm) -> Gaussian NB
        hp = [{"var_smoothing": v} for v in (1e-9, 1e-6, 1e-3, 1e-1)]
        base = lambda p: GaussianNB(**p)
        weighted = lambda p: GaussianNB(priors=[1 / 3] * 3, **p)
    elif model_name == "Naive Bayes":
        hp = [{"alpha": a} for a in (0.1, 0.3, 0.5, 1.0)]
        base = lambda p: MultinomialNB(**p)
        weighted = lambda p: MultinomialNB(fit_prior=False, **p)
    elif dense:  # vector dày 200 chiều -> SVM nhân RBF
        hp = [{"kernel": "rbf", "C": c} for c in (1, 10)]
        base = lambda p: SVC(**p, random_state=SEED)
        weighted = lambda p: SVC(**p, class_weight="balanced", random_state=SEED)
    else:
        hp = [{"kernel": "linear", "C": c} for c in (0.03, 0.1, 0.3, 1.0)]
        base = lambda p: LinearSVC(C=p["C"], random_state=SEED, max_iter=5000)
        weighted = lambda p: LinearSVC(C=p["C"], class_weight="balanced", random_state=SEED, max_iter=5000)

    grid = [("Không xử lý", None, p, lambda p=p: base(p)) for p in hp]
    grid += [("Trọng số lớp", None, p, lambda p=p: weighted(p)) for p in hp]
    grid += [(s, t, p, lambda p=p: base(p)) for s, t in sampling_grid(y_tr) for p in hp]
    return grid


def fit_one(make, targets, Xtr, y_tr, Xva, y_va):
    t0 = time.time()
    idx = resample_index(y_tr, targets) if targets else np.arange(len(y_tr))
    clf = make().fit(Xtr[idx], y_tr[idx])
    return f1_score(y_va, clf.predict(Xva), average="macro"), clf, time.time() - t0


def paired_bootstrap_delta(y, p_new, p_base, n_boot=1000, seed=SEED):
    """Chênh lệch Macro-F1 (p_new - p_base) và KTC 95% bằng bootstrap theo cặp trên tập test."""
    idx = np.random.default_rng(seed).integers(0, len(y), size=(n_boot, len(y)))
    d = np.array([f1_score(y[i], p_new[i], average="macro") - f1_score(y[i], p_base[i], average="macro") for i in idx])
    return float(np.percentile(d, 2.5)), float(np.percentile(d, 97.5))


def run_main_experiment(data):
    y_tr, y_va, y_te = (data[s]["label"].to_numpy() for s in ("train", "validation", "test"))
    print("phân phối train:", dict(zip(LABEL_NAMES, class_counts(y_tr).tolist())))
    feats = build_features(data)
    results, tuning, per_class, imbalance = [], [], {}, []

    for feat_name, (vec, Xtr, Xva, Xte) in feats.items():
        for model_name in ("Naive Bayes", "SVM"):
            grid = model_grid(model_name, feat_name, y_tr)
            fits = Parallel(n_jobs=N_JOBS)(delayed(fit_one)(make, t, Xtr, y_tr, Xva, y_va) for _, t, _, make in grid)
            best_by_strategy = {}
            for (strategy, targets, params, _), (f1_va, clf, fit_s) in zip(grid, fits):
                tuning.append({"feature": feat_name, "model": model_name, "strategy": strategy,
                               "sampling": describe(targets) if targets else "", **params,
                               "val_f1_macro": round(f1_va, 4), "fit_seconds": round(fit_s, 1)})
                if strategy not in best_by_strategy or f1_va > best_by_strategy[strategy][0]:
                    best_by_strategy[strategy] = (f1_va, targets, params, clf, fit_s)

            # So sánh các chiến lược: mỗi chiến lược lấy cấu hình tốt nhất trên validation
            key = f"{model_name} + {feat_name}"
            preds = {s: best_by_strategy[s][3].predict(Xte) for s in STRATEGIES}
            for s in STRATEGIES:
                f1_va, targets, params, _, _ = best_by_strategy[s]
                m = compute_metrics(y_te, preds[s])
                f1_cls = f1_score(y_te, preds[s], average=None)
                rec_cls = recall_score(y_te, preds[s], average=None)
                lo, hi = paired_bootstrap_delta(y_te, preds[s], preds["Không xử lý"]) if s != "Không xử lý" else (0.0, 0.0)
                imbalance.append({"model": key, "strategy": s, "params": params,
                                  "sampling": describe(targets) if targets else "",
                                  "val_f1_macro": f1_va, "accuracy": m["accuracy"], "f1_macro": m["f1_macro"],
                                  "f1_neutral": f1_cls[1], "recall_neutral": rec_cls[1],
                                  "f1_dissatisfied": f1_cls[0], "f1_satisfied": f1_cls[2],
                                  "delta_vs_none": m["f1_macro"] - compute_metrics(y_te, preds["Không xử lý"])["f1_macro"],
                                  "delta_ci_low": lo, "delta_ci_high": hi})
                print(f"  {key:22s} {s:16s} {params} {describe(targets) if targets else '':32s} "
                      f"val_f1={f1_va:.4f} test_f1={m['f1_macro']:.4f} f1_TT={f1_cls[1]:.3f}")

            # Mô hình cuối: chiến lược + siêu tham số có Macro-F1 validation cao nhất
            strategy = max(STRATEGIES, key=lambda s: best_by_strategy[s][0])
            f1_va, targets, params, clf, fit_s = best_by_strategy[strategy]
            y_pred = preds[strategy]
            m = compute_metrics(y_te, y_pred)
            print(f"> {key}: {strategy} {params} test_acc={m['accuracy']:.4f} test_f1={m['f1_macro']:.4f}")
            results.append({"model": model_name, "feature": feat_name, "best_params": params,
                            "imbalance_strategy": strategy, "sampling": describe(targets) if targets else "",
                            "val_f1_macro": f1_va, "fit_seconds": fit_s, "dim": Xtr.shape[1], **m})
            per_class[key] = classification_report(y_te, y_pred, target_names=LABEL_NAMES,
                                                   output_dict=True, zero_division=0)
            cm = confusion_matrix(y_te, y_pred)
            per_class[key]["confusion_matrix"] = cm.tolist()
            slug = key.replace(" + ", "_").replace(" ", "").replace("-", "").lower()
            plot_confusion(cm, key, FIG / f"cm_{slug}.png")
            joblib.dump({"vectorizer": vec, "model": clf}, MOD / f"{slug}.joblib")
    return results, tuning, per_class, imbalance


# ---------------------------------------------------------------- Thí nghiệm phụ: ảnh hưởng của tiền xử lý
def run_preprocessing_ablation(data):
    """TF-IDF + Linear SVM (C=0.3) trên các mức tiền xử lý khác nhau, với unigram và uni+bigram."""
    variants = {
        "Thô (chỉ lowercase)": "text_raw_lower",
        "Chuẩn hóa, chưa tách từ": "text_norm_clean",
        "Chuẩn hóa + tách từ": "tokens_all",
        "Chuẩn hóa + tách từ + bỏ stopword": "tokens",
    }
    for d in data.values():
        d["text_raw_lower"] = d["text"].str.lower()
        d["text_norm_clean"] = d["tokens_all"].str.replace("_", " ")  # bỏ ranh giới từ ghép
    y_tr, y_va, y_te = (data[s]["label"].to_numpy() for s in ("train", "validation", "test"))
    rows = []
    for (name, col), ngram in ((v, n) for n in ((1, 1), (1, 2)) for v in variants.items()):
        token_pattern = r"(?u)\b\w+\b" if col == "text_raw_lower" else r"\S+"
        vec = TfidfVectorizer(ngram_range=ngram, min_df=2, sublinear_tf=True, token_pattern=token_pattern)
        Xtr = vec.fit_transform(data["train"][col])
        clf = LinearSVC(C=0.3, class_weight="balanced", random_state=SEED, max_iter=5000).fit(Xtr, y_tr)
        m_va = f1_score(y_va, clf.predict(vec.transform(data["validation"][col])), average="macro")
        m = compute_metrics(y_te, clf.predict(vec.transform(data["test"][col])))
        rows.append({"variant": name, "ngram": f"{ngram[0]}-{ngram[1]}", "vocab_size": len(vec.vocabulary_), "val_f1_macro": m_va, **m})
        print(f"  ablation {name:35s} ngram={ngram} val_f1={m_va:.4f} vocab={len(vec.vocabulary_):6d} test_f1={m['f1_macro']:.4f}")
    return rows


def main():
    for d in (MET, FIG, MOD):
        d.mkdir(parents=True, exist_ok=True)
    data = load()

    results, tuning, per_class, imbalance = run_main_experiment(data)
    pd.DataFrame(tuning).to_csv(MET / "classical_tuning.csv", index=False)
    pd.DataFrame(imbalance).to_csv(MET / "imbalance_classical.csv", index=False)
    res_df = pd.DataFrame(results)
    res_df.to_csv(MET / "classical_results.csv", index=False)
    (MET / "classical_per_class.json").write_text(json.dumps(per_class, indent=2), encoding="utf-8")

    ablation = run_preprocessing_ablation(data)
    pd.DataFrame(ablation).to_csv(MET / "preprocessing_ablation.csv", index=False)

    feats = ["BoW", "TF-IDF", "Word2Vec"]
    models = ["Naive Bayes", "SVM"]
    for metric, label in (("f1_macro", "Macro-F1"), ("accuracy", "Accuracy")):
        vals = [[res_df.query("model == @m and feature == @f")[metric].item() for f in feats] for m in models]
        grouped_bars(feats, models, vals, f"{label} trên tập test theo đặc trưng và thuật toán",
                     label, FIG / f"compare_{metric}.png", ylim=(0, 1.05))


if __name__ == "__main__":
    main()
