"""Tổng hợp kết quả 7 mô hình: khoảng tin cậy bootstrap cho Macro-F1, F1 từng lớp, ví dụ lỗi."""
import json

import joblib
import numpy as np
import pandas as pd
from sklearn.metrics import f1_score

from plot_style import ci_dotplot, confusion_grid, grouped_bars
from text_utils import ROOT
from train_classical import LABEL_NAMES, STRATEGIES, MeanWord2Vec  # noqa: F401  (cần để joblib nạp Word2Vec)

FIG = ROOT / "outputs" / "figures"
PROC = ROOT / "outputs" / "processed"
MET = ROOT / "outputs" / "metrics"
MOD = ROOT / "outputs" / "models"
N_BOOT = 2000


def predict_classical(slug, texts):
    bundle = joblib.load(MOD / f"{slug}.joblib")
    vec = bundle["vectorizer"]
    if isinstance(vec, tuple):  # (Word2Vec, StandardScaler)
        X = vec[1].transform(vec[0].transform(texts))
    else:
        X = vec.transform(texts)
    return bundle["model"].predict(X)


def bootstrap_ci(y, preds, rng):
    """Bootstrap theo cặp: cùng một mẫu lặp cho mọi mô hình, để so sánh chênh lệch giữa hai mô hình."""
    n = len(y)
    idx = rng.integers(0, n, size=(N_BOOT, n))
    scores = {name: np.array([f1_score(y[i], p[i], average="macro") for i in idx]) for name, p in preds.items()}
    return scores


def main():
    test = pd.read_parquet(PROC / "test.parquet")
    y = test["label"].to_numpy()
    res = pd.read_csv(MET / "classical_results.csv")

    preds = {}
    for _, r in res.iterrows():
        slug = f"{r.model}_{r.feature}".replace(" ", "").replace("-", "").lower()
        preds[f"{r.model} + {r.feature}"] = predict_classical(slug, test["tokens_all"].tolist())
    bert = pd.read_csv(MET / "phobert_test_predictions.csv")
    assert (bert["id"].astype(str).to_numpy() == test["id"].astype(str).to_numpy()).all()
    preds["PhoBERT"] = bert["pred"].to_numpy()

    rng = np.random.default_rng(42)
    boot = bootstrap_ci(y, preds, rng)
    rows = []
    for name, p in preds.items():
        f1s = f1_score(y, p, average=None)
        lo, hi = np.percentile(boot[name], [2.5, 97.5])
        rows.append({"model": name, "f1_macro": f1_score(y, p, average="macro"), "ci_low": lo, "ci_high": hi,
                     **{f"f1_{c}": v for c, v in zip(LABEL_NAMES, f1s)}})
    summary = pd.DataFrame(rows).sort_values("f1_macro", ascending=False)
    summary.to_csv(MET / "all_models_summary.csv", index=False)
    print(summary.round(4).to_string(index=False))

    diff = boot["PhoBERT"] - boot["SVM + TF-IDF"]
    best_classical_vs_bert = {"mean_diff": float(diff.mean()), "ci": np.percentile(diff, [2.5, 97.5]).tolist(),
                              "p_bert_not_better": float((diff <= 0).mean())}
    diff2 = boot["SVM + TF-IDF"] - boot["SVM + BoW"]
    tfidf_vs_bow = {"mean_diff": float(diff2.mean()), "ci": np.percentile(diff2, [2.5, 97.5]).tolist()}
    print("PhoBERT - SVM+TF-IDF:", best_classical_vs_bert, "\nSVM TF-IDF - SVM BoW:", tfidf_vs_bow)

    s = summary.reset_index(drop=True)
    best_classical = s[s.model != "PhoBERT"].iloc[0]
    bert = s[s.model == "PhoBERT"].iloc[0]
    fmt = lambda v: f"{v:.3f}".replace(".", ",")
    ci_dotplot([m.replace("PhoBERT", "PhoBERT (fine-tune)") for m in s.model], s.f1_macro, s.ci_low, s.ci_high,
               f"PhoBERT dẫn đầu với Macro-F1 {fmt(bert.f1_macro)}, hơn {best_classical.model} "
               f"{fmt(round(bert.f1_macro, 3) - round(best_classical.f1_macro, 3))}",
               f"Macro-F1 trên tập test; thanh ngang = khoảng tin cậy 95% (bootstrap {N_BOOT:,} lần)".replace(",", "."),
               FIG / "compare_ci.png", ref=best_classical.f1_macro, ref_label="tốt nhất trong nhóm cổ điển ",
               highlight=int(s.index[s.model == "PhoBERT"][0]))

    # Phân tích lỗi: nhầm lẫn của PhoBERT và các câu PhoBERT đúng nhưng SVM+TF-IDF sai
    test = test.assign(svm=preds["SVM + TF-IDF"], bert=preds["PhoBERT"])
    name = dict(enumerate(LABEL_NAMES))
    bert_wrong = test[test.bert != test.label]
    pair_counts = bert_wrong.groupby(["label", "bert"]).size().sort_values(ascending=False)
    examples = {
        "phobert_error_pairs": {f"{name[a]}->{name[b]}": int(n) for (a, b), n in pair_counts.items()},
        "phobert_errors": bert_wrong.sample(25, random_state=1)[["text", "label", "bert"]].to_dict("records"),
        "bert_right_svm_wrong": test[(test.bert == test.label) & (test.svm != test.label)]
            .sample(20, random_state=1)[["text", "label", "svm"]].to_dict("records"),
        "n_bert_right_svm_wrong": int(((test.bert == test.label) & (test.svm != test.label)).sum()),
        "n_svm_right_bert_wrong": int(((test.svm == test.label) & (test.bert != test.label)).sum()),
        "phobert_minus_svm_tfidf": best_classical_vs_bert,
        "svm_tfidf_minus_svm_bow": tfidf_vs_bow,
    }
    (MET / "error_analysis.json").write_text(json.dumps(examples, ensure_ascii=False, indent=2), encoding="utf-8")

    imbalance_summary()

    per_class = json.loads((MET / "classical_per_class.json").read_text())
    order = [f"{m} + {f}" for m in ("Naive Bayes", "SVM") for f in ("BoW", "TF-IDF", "Word2Vec")]
    confusion_grid([per_class[k]["confusion_matrix"] for k in order], order, FIG / "cm_classical_grid.png")


def imbalance_summary():
    """Gộp bảng so sánh chiến lược xử lý mất cân bằng của 7 mô hình và vẽ biểu đồ."""
    df = pd.concat([pd.read_csv(MET / "imbalance_classical.csv"), pd.read_csv(MET / "imbalance_phobert.csv")])
    df.to_csv(MET / "imbalance_comparison.csv", index=False)
    groups = list(dict.fromkeys(df["model"]))
    for metric, label in (("f1_macro", "Macro-F1"), ("f1_neutral", "F1 lớp Trung tính")):
        vals = [[df.query("model == @g and strategy == @s")[metric].squeeze() if
                 ((df.model == g) & (df.strategy == s)).any() else np.nan for g in groups] for s in STRATEGIES]
        grouped_bars([g.replace(" + ", "\n") for g in groups], STRATEGIES, vals,
                     f"{label} trên tập test theo cách xử lý mất cân bằng", label,
                     FIG / f"imbalance_{metric}.png", ylim=(0, 1.0), labels=False, figwidth=8.5)
    print(df.pivot(index="model", columns="strategy", values="f1_macro").round(3).to_string())


if __name__ == "__main__":
    main()
