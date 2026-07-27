# SIC Face Recognition — FaceViT và Semi-Hard Triplet Loss

Dự án nhận diện khuôn mặt của nhóm trong khóa Samsung Innovation Campus (SIC). Phiên bản hiện tại tập trung vào việc tự xây dựng một Vision Transformer tạo face embedding, huấn luyện bằng Triplet Loss và đánh giá trên các identity chưa từng xuất hiện trong tập train.

Mô hình không dùng trọng số pretrained và không dùng classifier Softmax làm đầu ra chính.

## 1. Pipeline hiện tại

```text
Ảnh khuôn mặt RGB
    ↓
Resize và augmentation
    ↓
FaceViT
    ↓
Embedding 128 chiều, L2 norm = 1
    ↓
So sánh bằng Euclidean Distance
    ↓
Verification hoặc Identification
```

Hai bài toán đang được đánh giá:

- Verification: hai ảnh có phải cùng một người không?
- Identification: ảnh probe thuộc người nào trong gallery?

Face Detection nhiều khuôn mặt, Web App và Mobile/WinForms là các phase tiếp theo. Chúng chỉ nên được ghép vào sau khi face embedding đạt ROC-AUC và EER đủ tốt.

## 2. Cấu trúc project

```text
face_recognition_project/
├── README.md
├── requirements.txt
├── .gitignore
└── src/
    ├── config.py
    ├── data.py
    ├── model.py
    ├── train.py
    ├── test.py
    ├── metrics.py
    ├── visualization.py
    ├── dataset/
    │   └── vggface2/
    │       ├── train/
    │       └── val/
    ├── checkpoints/
    └── outputs/
```

`dataset`, `checkpoints` và `outputs` được loại khỏi Git bằng `.gitignore` vì có kích thước lớn hoặc được sinh tự động.

## 3. Dataset VGGFace2

Subset VGGFace2 đang sử dụng có cấu trúc:

```text
dataset/vggface2/
├── train/
│   ├── n000001/
│   ├── n000002/
│   └── ...
└── val/
    ├── n000009/
    ├── n000040/
    └── ...
```

Thống kê đã kiểm tra:

| Split nguồn | Identity | Ảnh |
|---|---:|---:|
| `train` | 480 | 176.398 |
| `val` | 60 | 21.295 |
| Tổng | 540 | 197.693 |

Identity trong `train` và `val` không trùng nhau. Code tiếp tục xáo trộn 60 identity holdout bằng `seed=42`, sau đó chia thành:

```text
Train:      480 identity, 176.398 ảnh
Validation:  30 identity,  10.957 ảnh
Test:        30 identity,  10.338 ảnh
```

Ba tập không trùng identity. Đây là cách đánh giá open-set: mô hình phải học đặc trưng khuôn mặt từ 480 người và áp dụng cho người chưa xuất hiện trong train.

## 4. Cài đặt

Thực hiện tại thư mục gốc project:

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
pip install -r requirements.txt
```

`requirements.txt` hiện mặc định cài PyTorch CUDA 12.4, phù hợp với NVIDIA RTX 4060. Kiểm tra GPU:

```powershell
python -c "import torch; print(torch.__version__); print(torch.cuda.is_available()); print(torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'CPU')"
```

Kết quả mong đợi:

```text
True
NVIDIA GeForce RTX 4060
```

Nếu muốn cài bản CPU, comment ba dòng CUDA và mở comment ba dòng CPU trong `requirements.txt`, sau đó cài lại PyTorch.

## 5. Kiểm tra dữ liệu

Các lệnh train/test phải chạy từ thư mục `src` vì đường dẫn dataset mặc định là `dataset/vggface2`:

```powershell
cd src
python data.py
```

Kết quả đúng:

```text
Number of classes: 540
Identity split: 480/30/30
PK batch images: torch.Size([16, 3, 224, 224])
PK batch labels: torch.Size([16])
```

Một batch train có bốn identity, mỗi identity bốn ảnh:

```text
P = 4 identity
K = 4 ảnh/identity
Batch = P × K = 16 ảnh
```

## 6. Huấn luyện

Chạy với cấu hình mặc định:

```powershell
python train.py
```

Lệnh tương đương:

```powershell
python train.py --epochs 100 --experiment_name sic_facevit_vggface2_semi_hard
```

Checkpoint tốt nhất được lưu tại:

```text
checkpoints/sic_facevit_vggface2_semi_hard_best.pth
```

Biểu đồ và lịch sử train được lưu tại:

```text
outputs/sic_facevit_vggface2_semi_hard/
├── history.json
└── training_curves.png
```

Không cần đợi đủ 100 epoch. Early Stopping tự dừng khi Validation Loss không cải thiện ít nhất `0.0001` trong 10 epoch liên tiếp.

## 7. Cấu hình chính

| Tham số | Mặc định | Ý nghĩa |
|---|---:|---|
| `image_size` | 224 | Kích thước ảnh đầu vào |
| `patch_size` | 16 | Mỗi patch là 16×16 pixel |
| `embed_dim` | 192 | Số chiều của mỗi token Transformer |
| `depth` | 12 | Số Transformer Encoder Block |
| `num_heads` | 3 | Số attention head |
| `mlp_ratio` | 4 | MLP 192 → 768 → 192 |
| `dropout` | 0.1 | Dropout chống overfitting |
| `face_embedding_dim` | 128 | Kích thước face embedding đầu ra |
| `identities_per_batch` | 4 | P trong P×K sampler |
| `images_per_identity` | 4 | K trong P×K sampler |
| `triplet_margin` | 0.2 | Khoảng cách an toàn giữa positive và negative |
| `lr` | 0.0003 | Learning rate của AdamW |
| `weight_decay` | 0.0001 | Regularization lên trọng số |
| `early_stop` | 10 | Patience của Early Stopping |
| `early_stop_min_delta` | 0.0001 | Mức cải thiện tối thiểu |
| `seed` | 42 | Tái lập cách chia và lấy mẫu |

Ví dụ thay đổi tham số:

```powershell
python train.py --lr 0.0001 --epochs 50 --experiment_name vggface2_lr_1e4
```

Chỉ nên đổi một nhóm tham số trong mỗi experiment để biết thay đổi nào tạo ra kết quả.

## 8. FaceViT hoạt động như thế nào?

Đầu vào:

```text
[B, 3, 224, 224]
```

### 8.1 Patch Embedding

`Conv2d(kernel_size=16, stride=16)` chia ảnh thành:

```text
224 / 16 = 14 patch mỗi chiều
14 × 14 = 196 patch
```

Mỗi patch được chiếu thành vector 192 chiều:

```text
[B, 3, 224, 224]
→ [B, 192, 14, 14]
→ [B, 196, 192]
```

### 8.2 CLS token và Position Embedding

Model thêm một CLS token để thu thập thông tin toàn ảnh:

```text
196 patch token + 1 CLS token = 197 token
[B, 197, 192]
```

Position Embedding cho model biết vị trí của từng patch. Một bộ Position Embedding được học và dùng chung cho mọi ảnh trong batch.

### 8.3 Transformer Encoder Block

Mỗi block gồm:

```text
LayerNorm
→ Multi-Head Self-Attention
→ Residual Connection
→ LayerNorm
→ FeedForward 192 → 768 → 192
→ Residual Connection
```

Self-Attention giúp mỗi patch xem thông tin từ toàn bộ patch còn lại. Với ba head:

```text
192 / 3 = 64 chiều/head
```

LayerNorm chuẩn hóa 192 đặc trưng của từng token, giúp giá trị ổn định hơn trong quá trình train. Residual Connection cộng đầu vào với đầu ra của attention/MLP để gradient truyền qua mạng sâu tốt hơn.

Pipeline có 12 Transformer Encoder Block liên tiếp.

### 8.4 Embedding Head

Sau block cuối:

```text
CLS [B, 192]
→ Linear(192, 128)
→ L2 Normalize
→ Face embedding [B, 128]
```

L2 Normalize bảo đảm mỗi embedding có độ dài bằng 1:

```text
||embedding||₂ = 1
```

Vì vậy Euclidean Distance giữa hai embedding nằm trong khoảng xấp xỉ `[0, 2]`.

## 9. P×K Sampler

Triplet Loss cần ít nhất hai ảnh cùng người và ảnh của người khác trong một batch. Sampler tạo:

```text
4 identity × 4 ảnh = 16 ảnh
```

Ví dụ:

```text
Identity A: A1 A2 A3 A4
Identity B: B1 B2 B3 B4
Identity C: C1 C2 C3 C4
Identity D: D1 D2 D3 D4
```

Train sampler tạo tổ hợp mới mỗi epoch. Validation sampler dùng cùng tổ hợp để Val Loss có thể so sánh ổn định giữa các epoch.

## 10. Semi-Hard Triplet Mining

Với mỗi Anchor, code tính ma trận khoảng cách giữa toàn bộ embedding trong batch:

```text
distances.shape = [16, 16]
```

Positive được chọn là ảnh cùng identity nhưng xa Anchor nhất. Đây là Hardest Positive.

Semi-Hard Negative phải thỏa mãn:

```text
PosDist < NegDist < PosDist + margin
```

Negative này khác người, xa hơn Positive nhưng chưa đủ xa để đạt margin. Nó tạo tín hiệu học hữu ích mà không quá cực đoan như Hardest Negative.

Nếu batch không có Semi-Hard Negative, code chọn Negative xa nhất làm fallback. Cách này hạn chế model bị embedding collapse khi mọi anchor bị ép học negative quá khó ngay từ đầu.

## 11. Triplet Margin Loss

Ba embedding:

```text
Anchor   = A
Positive = P, cùng người với A
Negative = N, khác người với A
```

Khoảng cách:

```text
PosDist = d(A, P)
NegDist = d(A, N)
```

Loss:

```text
Loss = max(PosDist - NegDist + margin, 0)
```

Với `margin=0.2`, model cần đạt:

```text
NegDist > PosDist + 0.2
```

Nếu điều kiện đã đạt, Loss của triplet đó bằng 0. Loss được in trên terminal là trung bình của toàn bộ triplet, vì vậy một số triplet bằng 0 không có nghĩa Loss trung bình cũng bằng 0.

## 12. Quy trình của một epoch

### Train

```text
model.train()
→ lấy P×K batch
→ tạo embedding
→ chọn Semi-Hard Triplet
→ tính TripletMarginLoss
→ loss.backward()
→ optimizer.step()
```

Trọng số được cập nhật sau mỗi batch. Train Loss của epoch là trung bình Loss trên toàn bộ batch train.

### Validation

```text
model.eval()
→ torch.no_grad()
→ tạo embedding
→ tính Val Loss
→ không backward
→ không cập nhật trọng số
```

Nếu Val Loss thấp hơn best loss ít nhất `min_delta`, chương trình lưu checkpoint mới. Nếu không cải thiện đủ trong 10 epoch, chương trình Early Stop.

## 13. Ý nghĩa log train

Ví dụ:

```text
Train Loss=0.1490 Val Loss=0.1510
PosDist=1.3200 NegDist=1.4600
TripletRate=16.00%
```

- Train Loss: Loss dùng để cập nhật trọng số.
- Val Loss: Loss trên identity validation, không cập nhật trọng số.
- PosDist: khoảng cách của ảnh cùng người, muốn thấp.
- NegDist: khoảng cách của ảnh khác người, muốn cao.
- TripletRate: tỷ lệ triplet đạt `NegDist > PosDist + margin`.

Dấu hiệu embedding collapse:

```text
Loss ≈ 0.2000
PosDist ≈ 0
NegDist ≈ 0
TripletRate = 0%
```

Nếu xuất hiện liên tục qua nhiều epoch, checkpoint đó không nên dùng.

## 14. Chạy test

Sau khi train hoàn tất:

```powershell
python test.py
```

Hoặc chỉ định experiment:

```powershell
python test.py --experiment_name sic_facevit_vggface2_semi_hard
```

Mặc định test tạo:

```text
10.000 verification pair
5 gallery image/identity
Các ảnh còn lại làm probe
```

Với 30 identity test, gallery có 150 ảnh. Mỗi probe được so sánh với toàn bộ gallery bằng Euclidean Distance.

## 15. Các metric đánh giá

### Mean Positive/Negative Distance

- Mean Positive Distance: khoảng cách trung bình của cặp cùng người, muốn thấp.
- Mean Negative Distance: khoảng cách trung bình của cặp khác người, muốn cao.

### ROC-AUC

Đo khả năng xếp cặp cùng người gần hơn cặp khác người:

```text
0.50: tương đương ngẫu nhiên
1.00: phân biệt hoàn hảo
```

### EER

Equal Error Rate là điểm mà False Acceptance Rate và False Rejection Rate gần bằng nhau. EER càng thấp càng tốt.

`EER distance threshold` có thể dùng làm ngưỡng ban đầu cho ứng dụng:

```text
distance <= threshold → cùng người
distance > threshold  → khác người/Unknown
```

### TAR@FAR

- TAR: tỷ lệ chấp nhận đúng người thật.
- FAR: tỷ lệ chấp nhận nhầm người khác.

Project báo cáo:

```text
TAR@FAR=1%
TAR@FAR=0.1%
```

Đây là metric quan trọng cho hệ thống kiểm soát ra vào hoặc điểm danh.

### Recall@K

- Recall@1: identity đúng đứng ở vị trí đầu tiên.
- Recall@5: identity đúng xuất hiện trong năm kết quả gần nhất.

### mAP

Mean Average Precision đánh giá toàn bộ thứ tự gallery, không chỉ kết quả đầu tiên. mAP càng gần 100% càng tốt.

## 16. Output sau test

```text
outputs/sic_facevit_vggface2_semi_hard/
├── test_results.json
├── test_results.png
└── test_embeddings.pt
```

`test_results.json` chứa toàn bộ số liệu để đưa vào báo cáo.

`test_results.png` gồm:

1. Phân bố khoảng cách cùng người/khác người và EER threshold.
2. ROC Curve.
3. Recall@1, Recall@5 và mAP.

`test_embeddings.pt` chứa:

- Embedding 128 chiều.
- Label và đường dẫn ảnh.
- Gallery/probe indices.
- Danh sách identity.
- EER threshold.

File này sẽ được dùng lại khi phát triển demo nhận diện ảnh hoặc webcam.

## 17. Vai trò của từng file

### `config.py`

Khai báo toàn bộ tham số dòng lệnh và kiểm tra cấu hình hợp lệ.

### `data.py`

Đọc VGGFace2, chia 480/30/30 identity, áp dụng augmentation, tạo `FaceImageDataset`, P×K sampler và DataLoader.

### `model.py`

Định nghĩa Patch Embedding, FeedForward, Transformer Encoder Block và FaceVisionTransformer.

### `train.py`

Thiết lập seed/device, Semi-Hard Mining, Triplet Loss, train/validation loop, Early Stopping, checkpoint và lịch sử huấn luyện.

### `metrics.py`

Tính ROC-AUC, EER, threshold, TAR@FAR, Recall@K và mAP.

### `visualization.py`

Vẽ biểu đồ train và test, lưu history/results dưới dạng JSON.

### `test.py`

Nạp best checkpoint, trích xuất embedding, tạo verification pair, chia gallery/probe, tính metric và lưu artifact phục vụ ứng dụng.

## 18. Cơ sở học thuật

Kiến trúc là mô hình custom kết hợp các ý tưởng:

- Vision Transformer: Dosovitskiy et al., *An Image is Worth 16×16 Words*, ICLR 2021.
- DeiT-Tiny-style configuration: Touvron et al., *Training Data-Efficient Image Transformers*, ICML 2021.
- Triplet embedding và Semi-Hard Negative Mining: Schroff et al., *FaceNet*, CVPR 2015.

Tên mô hình trong project:

```text
SIC FaceViT — ViT-Tiny/16-style backbone
+ FaceNet-style Semi-Hard Triplet training
```

Đây không phải bản sao nguyên vẹn của một mô hình trong bài báo và không dùng pretrained weights.

## 19. Kết quả thí nghiệm

Các kết quả Pins Face Recognition trước khi chuyển sang VGGFace2:

| Phương pháp | Kết quả |
|---|---|
| Random Triplet | ROC-AUC 0.5231, EER 48.48% |
| Batch-Hard từ đầu | Embedding collapse |
| Semi-Hard | Best Val Loss 0.1480 |

Kết quả VGGFace2 sẽ được bổ sung sau khi quá trình train và `test.py` hoàn tất.

## 20. Git workflow

Sau khi sửa code:

```powershell
git status
git add .
git commit -m "mo ta thay doi"
git push origin main
```

Trên máy train:

```powershell
git pull origin main
```

Không commit dataset, checkpoint hoặc output lên GitHub.
