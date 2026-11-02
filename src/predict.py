"""Dự đoán cảm xúc cho câu bất kỳ bằng PhoBERT và mô hình cổ điển tốt nhất (SVM + TF-IDF)."""
import sys

import joblib
import torch
from transformers import AutoModelForSequenceClassification, AutoTokenizer

from text_utils import ROOT, clean_tokens, normalize, segment
from train_classical import MeanWord2Vec  # noqa: F401  (cần để joblib nạp mô hình Word2Vec)

LABELS_VI = ["Không hài lòng", "Trung tính", "Hài lòng"]
MODELS = ROOT / "outputs" / "models"


def main(texts):
    svm = joblib.load(MODELS / "svm_tfidf.joblib")
    tok = AutoTokenizer.from_pretrained(MODELS / "phobert" / "best")
    bert = AutoModelForSequenceClassification.from_pretrained(MODELS / "phobert" / "best").eval()

    for text in texts:
        seg = segment(normalize(text))
        svm_pred = svm["model"].predict(svm["vectorizer"].transform([" ".join(clean_tokens(seg))]))[0]
        with torch.no_grad():
            probs = bert(**tok(seg, return_tensors="pt", truncation=True, max_length=256)).logits.softmax(-1)[0]
        print(f"\n{text}\n  SVM + TF-IDF : {LABELS_VI[svm_pred]}")
        print(f"  PhoBERT      : {LABELS_VI[int(probs.argmax())]}  "
              + "  ".join(f"{l}={p:.2f}" for l, p in zip(LABELS_VI, probs.tolist())))


if __name__ == "__main__":
    main(sys.argv[1:] or [
        "Máy giặt chạy êm, nhân viên giao hàng nhiệt tình",
        "Dùng được 2 tuần là hỏng, bảo hành thì hẹn mãi không tới",
        "Tạm được, giá hơi cao so với chất lượng",
        "Quạt không ồn như mình nghĩ, mát lắm",
    ])
