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
```

Tùy chọn: sửa lỗi chính tả cả 3 tập train/validation/test bằng ViT5 rồi huấn luyện lại toàn bộ mô hình.
Cờ `--vit5` ghi mọi kết quả vào `outputs/vit5/`, không đụng tới kết quả gốc trong `outputs/`:

```bash
python preprocess.py --vit5       # ViT5 sửa 3 tập (~1 giờ GPU, có cache) rồi tiền xử lý
python train_classical.py --vit5
python train_phobert.py --vit5
python analysis.py --vit5
python vit5_correction.py         # so sánh dữ liệu gốc với dữ liệu đã sửa bằng ViT5
```

Dự đoán thử một câu (cần chạy xong `train_classical.py` và `train_phobert.py`):

```bash
python predict.py "Máy giặt chạy êm, giao hàng nhanh"
```

## Kết quả

- `outputs/metrics/`: bảng kết quả và log
- `outputs/figures/`: biểu đồ
- `outputs/models/`: mô hình đã huấn luyện (tạo ra khi chạy, không có trong repo)
- `outputs/vit5/`: các kết quả trên của thí nghiệm sửa chính tả bằng ViT5
