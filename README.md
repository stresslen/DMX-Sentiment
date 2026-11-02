# Phân loại cảm xúc bình luận Điện Máy Xanh (DMX-Sentiment)

Bước 2–5 và mô hình hiện đại (ý 4) cho bài toán phân loại 3 lớp: Không hài lòng / Trung tính / Hài lòng.

## Cấu trúc

```
DMX-Sentiment-v1.zip          bộ dữ liệu (bước 1), giải nén vào data/
resources/vietnamese_stopwords.txt   danh sách stopword (giữ lại từ phủ định, mức độ)
src/text_utils.py             chuẩn hóa, teencode, tách từ (pyvi), loại stopword
src/imbalance.py              xử lý mất cân bằng: under/over/hybrid sampling, trọng số lớp
src/preprocess.py             Bước 2  -> outputs/processed/*.parquet
src/train_classical.py        Bước 3-5: BoW / TF-IDF / Word2Vec × Naive Bayes / SVM × 5 cách xử lý mất cân bằng
                              + thí nghiệm tiền xử lý
src/train_phobert.py          Ý 4: fine-tune vinai/phobert-base-v2 (không xử lý / trọng số lớp / oversampling)
src/analysis.py               Tổng hợp 7 mô hình: bootstrap CI, F1 từng lớp, phân tích lỗi, so sánh xử lý mất cân bằng
src/vit5_test_correction.py    Thí nghiệm: sửa lỗi tập test bằng ViT5 (ProtonX legal-tc) rồi đánh giá lại 7 mô hình
src/predict.py                Dự đoán thử cho câu bất kỳ
src/plot_style.py             Màu và hàm vẽ biểu đồ
outputs/metrics/              bảng kết quả (csv/json), log
outputs/figures/              ma trận nhầm lẫn, biểu đồ so sánh, đường huấn luyện
outputs/models/               mô hình đã huấn luyện (joblib, PhoBERT checkpoint)
```

## Chạy lại

Repo không kèm bộ dữ liệu (`DMX-Sentiment-v1.zip`), `outputs/processed/` và `outputs/models/`; đặt file zip vào thư mục gốc rồi chạy các bước dưới để tạo lại.

```bash
python3 -m venv --system-site-packages .venv      # dùng lại torch/sklearn có sẵn
.venv/bin/pip install -r requirements.txt
unzip DMX-Sentiment-v1.zip -d data

cd src
../.venv/bin/python preprocess.py        # ~5 giây
../.venv/bin/python train_classical.py   # ~15 phút (CPU, SVM-RBF chiếm phần lớn)
../.venv/bin/python train_phobert.py     # ~30 phút trên RTX 3060 12GB (3 lần fine-tune)
../.venv/bin/python analysis.py          # ~1 phút
../.venv/bin/python vit5_test_correction.py   # ~8 phút (GPU), tùy chọn
../.venv/bin/python predict.py "Máy giặt chạy êm, giao hàng nhanh"
```

## Xử lý mất cân bằng dữ liệu

Tập train lệch mạnh: Hài lòng 33.363 (71,7%), Không hài lòng 11.766 (25,3%), Trung tính 1.373 (3,0%).
Chỉ tập train được lấy mẫu lại; validation và test giữ phân phối thật.

| Chiến lược | Naive Bayes | SVM | PhoBERT |
|---|---|---|---|
| Không xử lý | prior theo tần suất | không trọng số | cross-entropy thường |
| Trọng số lớp | prior đều (`fit_prior=False` / `priors` đều) | `class_weight="balanced"` | cross-entropy trọng số √(N / (K·N_c)) |
| Undersampling | giảm Hài lòng xuống 20.000 / 12.000 | như NB | – |
| Oversampling | tăng Trung tính lên 4.000 / 8.000 / 12.000 | như NB | Trung tính → 4.000 |
| Hybrid sampling | kết hợp hai lưới trên | như NB | – |

Chiến lược và siêu tham số được chọn theo Macro-F1 trên tập validation; tập test chỉ dùng để báo cáo kết quả cuối. Seed = 42.
Bảng so sánh: `outputs/metrics/imbalance_comparison.csv`, hình `outputs/figures/imbalance_*.png`.
