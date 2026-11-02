"""Bổ sung ý 4: mô hình hiện đại - fine-tune PhoBERT-base-v2 cho phân loại 3 lớp cảm xúc.

Đầu vào là văn bản đã chuẩn hóa + tách từ (cột text_seg), đúng định dạng PhoBERT được tiền huấn luyện.
Thử 3 cách xử lý mất cân bằng (không xử lý / trọng số lớp trong loss / oversampling lớp Trung tính).
Chọn checkpoint và chiến lược theo Macro-F1 trên validation, đánh giá trên test.
"""
import json
import shutil
import sys
import time

import numpy as np
import pandas as pd
import torch
from datasets import Dataset
from sklearn.metrics import classification_report, confusion_matrix, f1_score, recall_score
from transformers import (AutoModelForSequenceClassification, AutoTokenizer, DataCollatorWithPadding,
                          Trainer, TrainingArguments, set_seed)

from plot_style import SERIES, TEXT, TEXT_2, plot_confusion, plt
from text_utils import ROOT
from imbalance import NEUTRAL, describe, resample_index, sqrt_inverse_weights
from train_classical import LABEL_NAMES, compute_metrics, paired_bootstrap_delta

MODEL_NAME = "vinai/phobert-base-v2"
MAX_LEN = 256
SEED = 42
PROC = ROOT / "outputs" / "processed"
MET = ROOT / "outputs" / "metrics"
FIG = ROOT / "outputs" / "figures"
CKPT = ROOT / "outputs" / "models" / "phobert"


class WeightedTrainer(Trainer):
    """Cross-entropy có trọng số lớp (căn bậc hai nghịch đảo tần suất) để bù lớp Trung tính hiếm.

    class_weights = None -> cross-entropy thường (không xử lý mất cân bằng)."""

    def __init__(self, class_weights, **kw):
        super().__init__(**kw)
        self.class_weights = class_weights

    def compute_loss(self, model, inputs, return_outputs=False, **kw):
        labels = inputs.pop("labels")
        outputs = model(**inputs)
        weight = None if self.class_weights is None else self.class_weights.to(outputs.logits.device)
        loss = torch.nn.functional.cross_entropy(outputs.logits, labels, weight=weight)
        return (loss, outputs) if return_outputs else loss


# Chiến lược xử lý mất cân bằng: tên -> (dùng trọng số lớp trong loss, số mẫu mục tiêu khi lấy mẫu lại)
STRATEGIES = {
    "Không xử lý": (False, None),
    "Trọng số lớp": (True, None),                 # cross-entropy có trọng số căn bậc hai nghịch đảo tần suất
    "Oversampling": (False, {NEUTRAL: 4000}),     # nhân bản ngẫu nhiên lớp Trung tính 1.373 -> 4.000 mẫu
}


def main():
    for d in (MET, FIG, CKPT):
        d.mkdir(parents=True, exist_ok=True)
    tok = AutoTokenizer.from_pretrained(MODEL_NAME)
    splits = {s: pd.read_parquet(PROC / f"{s}.parquet") for s in ("train", "validation", "test")}

    def encode(df):
        ds = Dataset.from_pandas(df[["text_seg", "label"]].rename(columns={"label": "labels"}), preserve_index=False)
        return ds.map(lambda b: tok(b["text_seg"], truncation=True, max_length=MAX_LEN), batched=True,
                      remove_columns=["text_seg"])

    ds = {s: encode(df) for s, df in splits.items()}
    lens = np.array([len(x) for x in ds["train"]["input_ids"]])
    print(f"subword length: mean={lens.mean():.1f} p95={np.percentile(lens, 95):.0f} "
          f"truncated={np.mean(lens >= MAX_LEN):.2%}")

    y_tr, y_true = splits["train"]["label"].to_numpy(), splits["test"]["label"].to_numpy()
    weights = torch.tensor(sqrt_inverse_weights(y_tr), dtype=torch.float)
    print("class weights:", weights.tolist())
    evaluator = lambda model: Trainer(  # group_by_length=False để dự đoán giữ đúng thứ tự dòng của tập test
        model=model, data_collator=DataCollatorWithPadding(tok),
        args=TrainingArguments(output_dir=str(CKPT / "eval"), per_device_eval_batch_size=128,
                               fp16=torch.cuda.is_available(), report_to="none"))

    if "--eval-only" in sys.argv:  # đánh giá lại checkpoint tốt nhất đã lưu, không huấn luyện lại
        old = json.loads((MET / "phobert_results.json").read_text())
        model = AutoModelForSequenceClassification.from_pretrained(CKPT / "best")
        best = {"strategy": old["summary"]["imbalance_strategy"], "history": old["log_history"],
                "fit_seconds": old["summary"]["fit_seconds"]}
        best.update(evaluate(evaluator(model), ds))
        rows = None
    else:
        rows, best, preds = [], None, {}
        for strategy, (use_weights, targets) in STRATEGIES.items():
            set_seed(SEED)
            train_ds = ds["train"].select(resample_index(y_tr, targets)) if targets else ds["train"]
            print(f"=== {strategy}: {len(train_ds)} mẫu train, phân phối "
                  f"{np.bincount(np.asarray(train_ds['labels']), minlength=3).tolist()}")
            model, history, train_seconds = train(tok, train_ds, ds["validation"], weights if use_weights else None)
            out = evaluate(evaluator(model), ds)
            preds[strategy] = out["y_pred"]
            f1_cls = f1_score(y_true, out["y_pred"], average=None)
            rows.append({"model": "PhoBERT", "strategy": strategy,
                         "params": {"class_weights": "sqrt-inverse" if use_weights else None,
                                    "neutral_oversample_to": targets and targets[NEUTRAL]},
                         "sampling": describe(dict(enumerate(np.bincount(np.asarray(train_ds["labels"]), minlength=3))))
                         if targets else "",
                         "val_f1_macro": out["val_f1"], "fit_seconds": train_seconds,
                         **{k: v for k, v in compute_metrics(y_true, out["y_pred"]).items() if k in ("accuracy", "f1_macro")},
                         "f1_neutral": f1_cls[1], "recall_neutral": recall_score(y_true, out["y_pred"], average=None)[1],
                         "f1_dissatisfied": f1_cls[0], "f1_satisfied": f1_cls[2]})
            print({k: v for k, v in rows[-1].items() if k != "params"})
            if best is None or out["val_f1"] > best["val_f1"]:  # giữ mô hình có Macro-F1 validation cao nhất
                model.save_pretrained(str(CKPT / "best"))
                tok.save_pretrained(str(CKPT / "best"))
                best = {"strategy": strategy, "history": history, "fit_seconds": train_seconds, **out}
            del model
            torch.cuda.empty_cache()
        base = preds["Không xử lý"]
        for r in rows:
            r["delta_vs_none"] = r["f1_macro"] - compute_metrics(y_true, base)["f1_macro"]
            r["delta_ci_low"], r["delta_ci_high"] = (paired_bootstrap_delta(y_true, preds[r["strategy"]], base)
                                                     if r["strategy"] != "Không xử lý" else (0.0, 0.0))
        pd.DataFrame(rows).to_csv(MET / "imbalance_phobert.csv", index=False)

    y_pred = best["y_pred"]
    use_weights, targets = STRATEGIES[best["strategy"]]
    m = compute_metrics(y_true, y_pred)
    result = {"model": "PhoBERT-base-v2 (fine-tune)", "feature": "Contextual embeddings", "val_f1_macro": best["val_f1"],
              "fit_seconds": best["fit_seconds"], "test_inference_seconds": best["infer_seconds"],
              "imbalance_strategy": best["strategy"],
              "best_params": {"lr": 2e-5, "epochs": 4, "batch": 32, "max_len": MAX_LEN,
                              "class_weights": "sqrt-inverse" if use_weights else None,
                              "neutral_oversample_to": targets and targets[NEUTRAL]},
              **m}
    report = classification_report(y_true, y_pred, target_names=LABEL_NAMES, output_dict=True, zero_division=0)
    cm = confusion_matrix(y_true, y_pred)
    report["confusion_matrix"] = cm.tolist()
    (MET / "phobert_results.json").write_text(json.dumps({"summary": result, "per_class": report,
                                                          "log_history": best["history"]}, indent=2))
    out = splits["test"][["id", "text", "label"]].copy()
    out["pred"] = y_pred
    out["prob_" + pd.Index(LABEL_NAMES)] = best["probs"]
    out.to_csv(MET / "phobert_test_predictions.csv", index=False, encoding="utf-8-sig")
    plot_confusion(cm, "PhoBERT-base-v2", FIG / "cm_phobert.png")
    plot_training_curve(best["history"], FIG / "phobert_training.png")
    print(json.dumps(result, indent=2, ensure_ascii=False))


def evaluate(evaluator, ds):
    val_out = evaluator.predict(ds["validation"])
    t0 = time.time()
    test_out = evaluator.predict(ds["test"])
    infer_seconds = time.time() - t0
    assert (test_out.label_ids == np.asarray(ds["test"]["labels"])).all(), "dự đoán bị lệch thứ tự so với tập test"
    return {"val_f1": f1_score(val_out.label_ids, val_out.predictions.argmax(-1), average="macro"),
            "y_pred": test_out.predictions.argmax(-1), "infer_seconds": infer_seconds,
            "probs": torch.tensor(test_out.predictions).float().softmax(-1).numpy()}


def train(tok, train_ds, val_ds, weights):
    model = AutoModelForSequenceClassification.from_pretrained(
        MODEL_NAME, num_labels=3, id2label=dict(enumerate(LABEL_NAMES)),
        label2id={n: i for i, n in enumerate(LABEL_NAMES)})

    shutil.rmtree(CKPT / "runs", ignore_errors=True)  # checkpoint trung gian của lần chạy trước
    args = TrainingArguments(
        output_dir=str(CKPT / "runs"), seed=SEED,
        num_train_epochs=4, learning_rate=2e-5, weight_decay=0.01, warmup_ratio=0.1,
        per_device_train_batch_size=32, per_device_eval_batch_size=128,
        fp16=torch.cuda.is_available(), group_by_length=True,
        eval_strategy="epoch", save_strategy="epoch", save_total_limit=1,
        load_best_model_at_end=True, metric_for_best_model="f1_macro", greater_is_better=True,
        logging_steps=100, report_to="none", dataloader_num_workers=4,
    )
    trainer = WeightedTrainer(
        class_weights=weights, model=model, args=args,
        train_dataset=train_ds, eval_dataset=val_ds,
        data_collator=DataCollatorWithPadding(tok),
        compute_metrics=lambda p: {"f1_macro": f1_score(p.label_ids, p.predictions.argmax(-1), average="macro")},
    )
    t0 = time.time()
    trainer.train()  # load_best_model_at_end -> trainer.model là checkpoint có Macro-F1 validation cao nhất
    return trainer.model, trainer.state.log_history, time.time() - t0


def plot_training_curve(history, path):
    train = [(h["epoch"], h["loss"]) for h in history if "loss" in h]
    val = list({h["epoch"]: (h["epoch"], h["eval_loss"], h["eval_f1_macro"])
                for h in history if "eval_f1_macro" in h}.values())
    fig, (a1, a2) = plt.subplots(1, 2, figsize=(9, 3.2))
    a1.plot(*zip(*train), color=SERIES[0], lw=2, label="Train loss")
    a1.plot([v[0] for v in val], [v[1] for v in val], color=SERIES[1], lw=2, marker="o", ms=6, label="Validation loss")
    a1.set_xlabel("Epoch"); a1.set_title("Hàm mất mát", loc="left", fontsize=10.5, color=TEXT)
    a1.legend(fontsize=8.5); a1.grid(axis="y")
    a2.plot([v[0] for v in val], [v[2] for v in val], color=SERIES[0], lw=2, marker="o", ms=6)
    for e, _, f in val:
        a2.annotate(f"{f:.3f}", (e, f), textcoords="offset points", xytext=(0, 7), ha="center", fontsize=8, color=TEXT_2)
    a2.set_xlabel("Epoch"); a2.set_title("Macro-F1 trên validation", loc="left", fontsize=10.5, color=TEXT)
    a2.grid(axis="y")
    fig.tight_layout()
    fig.savefig(path)
    plt.close(fig)


if __name__ == "__main__":
    main()
