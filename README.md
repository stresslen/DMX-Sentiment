# DMX-Sentiment

Phân loại cảm xúc bình luận Điện Máy Xanh thành 3 lớp: Không hài lòng / Trung tính / Hài lòng.

## Yêu cầu

- Python 3.10 trở lên
- GPU NVIDIA (khuyến nghị, cho PhoBERT và ViT5)
- File dữ liệu `DMX-Sentiment-v1.zip` (không kèm trong repo)

## Cài đặt

```bash
git clone git@github.com:stresslen/DMX-Sentiment.git
cd DMX-Sentiment

python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

Chép `DMX-Sentiment-v1.zip` vào thư mục gốc của repo rồi giải nén:

```bash
unzip DMX-Sentiment-v1.zip -d data
```

## Chạy

Chạy lần lượt trong thư mục `src/`:

```bash
cd src
python preprocess.py              # tiền xử lý dữ liệu (~5 giây)
python train_classical.py         # Naive Bayes, SVM (~15 phút, CPU)
python train_phobert.py           # fine-tune PhoBERT (~30 phút, GPU)
python analysis.py                # tổng hợp, so sánh mô hình (~1 phút)
python vit5_test_correction.py    # tùy chọn: sửa lỗi tập test bằng ViT5 (~8 phút, GPU)
```

Dự đoán thử một câu (cần chạy xong `train_classical.py` và `train_phobert.py`):

```bash
python predict.py "Máy giặt chạy êm, giao hàng nhanh"
```

## Kết quả

- `outputs/metrics/`: bảng kết quả và log
- `outputs/figures/`: biểu đồ
- `outputs/models/`: mô hình đã huấn luyện (tạo ra khi chạy, không có trong repo)
