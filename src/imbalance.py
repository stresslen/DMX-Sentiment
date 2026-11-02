"""Các phương pháp cơ bản xử lý mất cân bằng dữ liệu: lấy mẫu lại (undersampling / oversampling / hybrid) và trọng số lớp.

Chỉ áp dụng trên tập train; validation và test giữ nguyên phân phối thật.
Nhãn: 0 = Không hài lòng, 1 = Trung tính, 2 = Hài lòng (lớp đa số).
"""
import numpy as np

SEED = 42
DISSATISFIED, NEUTRAL, SATISFIED = 0, 1, 2

# Lưới số mẫu mục tiêu sau khi lấy mẫu lại: giảm lớp Hài lòng (đa số) và/hoặc tăng lớp Trung tính (thiểu số).
# None = giữ nguyên số mẫu gốc của lớp đó. Lớp Không hài lòng (11.766 mẫu) luôn giữ nguyên.
SATISFIED_TARGETS = (None, 20000, 12000)
NEUTRAL_TARGETS = (None, 4000, 8000, 12000)


def class_counts(y):
    return np.bincount(y, minlength=3)


def resample_index(y, targets, seed=SEED):
    """Chỉ số các mẫu train sau khi lấy mẫu lại ngẫu nhiên đến đúng số lượng `targets[c]` cho mỗi lớp c.

    Lớp nhiều hơn mục tiêu -> random undersampling (bỏ bớt ngẫu nhiên, không lặp).
    Lớp ít hơn mục tiêu   -> random oversampling (giữ toàn bộ mẫu gốc, nhân bản ngẫu nhiên phần còn thiếu).
    """
    rng = np.random.default_rng(seed)
    idx = []
    for c in range(3):
        cls = np.flatnonzero(y == c)
        n = targets.get(c) or len(cls)
        if n <= len(cls):
            idx.append(rng.choice(cls, n, replace=False))
        else:
            idx.append(np.concatenate([cls, rng.choice(cls, n - len(cls), replace=True)]))
    return rng.permutation(np.concatenate(idx))


def sampling_grid(y):
    """Các cấu hình lấy mẫu lại cần thử: (tên chiến lược, {lớp: số mẫu})."""
    n = class_counts(y)
    grid = []
    for n_sat in SATISFIED_TARGETS:
        for n_neu in NEUTRAL_TARGETS:
            under, over = n_sat is not None, n_neu is not None
            if not (under or over):
                continue
            strategy = "Hybrid sampling" if under and over else "Undersampling" if under else "Oversampling"
            grid.append((strategy, {DISSATISFIED: int(n[DISSATISFIED]),
                                    NEUTRAL: n_neu or int(n[NEUTRAL]),
                                    SATISFIED: n_sat or int(n[SATISFIED])}))
    return grid


def describe(targets):
    return f"KHL={targets[DISSATISFIED]} / TT={targets[NEUTRAL]} / HL={targets[SATISFIED]}"


def balanced_weights(y):
    """Trọng số lớp kiểu class_weight='balanced' của scikit-learn: w_c = N / (K * N_c)."""
    n = class_counts(y)
    return n.sum() / (len(n) * n)


def sqrt_inverse_weights(y):
    """Căn bậc hai của trọng số balanced: bù lớp hiếm nhưng nhẹ tay hơn, tránh dự đoán thừa lớp hiếm."""
    return np.sqrt(balanced_weights(y))
