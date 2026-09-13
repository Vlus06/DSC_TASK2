# LegalQA Task 2 — Hệ thống hỏi đáp pháp luật tiếng Việt

Project xây dựng hệ thống tìm kiếm và sinh câu trả lời cho câu hỏi pháp luật tiếng Việt trong UIT Data Science Challenge — LegalQA Task 2.

Hệ thống kết hợp tìm kiếm từ khóa, tìm kiếm ngữ nghĩa, chia văn bản pháp luật theo cấu trúc, reranking bằng mô hình ngôn ngữ và Learning to Rank. Toàn bộ pipeline có thể chạy trên máy cá nhân hoặc Modal, đồng thời tự quản lý cache để hạn chế xử lý lại dữ liệu và tiết kiệm chi phí GPU.

## 1. Các thành phần chính

Project triển khai các phần sau:

- tiền xử lý 8.532 văn bản pháp luật;
- chia văn bản thành parent chunk và child chunk;
- lập chỉ mục BM25 cho tìm kiếm từ khóa;
- tạo embedding cho toàn bộ parent và child chunk;
- kết hợp BM25 với dense retrieval để chọn tài liệu liên quan;
- rerank các đoạn ứng viên bằng Vietnamese CrossEncoder;
- xây dựng 40 đặc trưng cho từng phương án trả lời;
- huấn luyện ensemble gồm năm mô hình XGBRanker;
- sinh câu trả lời có trích dẫn tên và đường dẫn văn bản;
- chạy theo từng giai đoạn trên CPU/H100 và lưu cache trên Modal Volume;
- lưu log, checkpoint, model và file dự đoán cho mỗi lần chạy.

## 2. Kiến trúc pipeline

```text
Câu hỏi
   │
   ├── BM25 top 400
   └── Dense document top 250
             │
             ▼
   Weighted score fusion (0.5 BM25 + 1.0 dense)
             │
             ▼
      Top 30 tài liệu → lấy top 3
             │
             ▼
   Candidate pool từ parent chunk + child chunk
             │
             ▼
      Dense similarity top 10 chunk
             │
             ▼
   Vietnamese_Reranker → CrossEncoder top 5
             │
             ▼
   5 singleton + 10 cặp không thứ tự = 15 actions
             │
             ▼
         ranking features
             │
             ▼
   XGBRanker × 5 seed → mean rank theo từng qid
             │
             ▼
       Answer builder, tối đa 5.800 ký tự
```

Hai mô hình ngôn ngữ được sử dụng:

- bi-encoder: `AITeamVN/Vietnamese_Embedding`;
- cross-encoder: `AITeamVN/Vietnamese_Reranker`.

Bộ chọn cuối gồm năm `XGBRanker` với seed `42`, `10042`, `20042`, `30042`, `40042`. Mỗi model dùng 350 cây, learning rate `0.03`, max depth `4`, min child weight `5`, subsample và colsample `0.9`, lambda `2.0`, `tree_method="hist"`.

## 3. Cấu trúc repository

```text
DSC_TASK2/
├── data/
│   ├── TASK2/
│   │   ├── train.json
│   │   ├── public-official.json
│   │   └── selected-contexts/
│   ├── cache/
│   └── stopwords.txt
├── outputs/
├── scripts/
│   ├── run.py                 # entrypoint local
│   ├── pipeline.py            # điều phối pipeline chính
│   ├── build_bm25.py
│   ├── build_dense_parent.py
│   ├── build_child.py
│   └── modal_stages.py    # các stage chạy trên Modal
├── src/legalqa/               # retrieval, feature, model và answer builder
├── modal_app.py               # workflow Modal từ dữ liệu đến inference
├── requirements.txt
└── README.md
```

## 4. Dữ liệu và cache

Các đầu vào bắt buộc:

```text
data/TASK2/train.json
data/TASK2/public-official.json
data/TASK2/selected-contexts/
data/stopwords.txt
```

Ba cache nền được lưu trong `data/cache/`:

| Cache | Nội dung | Cách xử lý trên Modal |
|---|---|---|
| `bm25_index.pkl` | Chỉ mục BM25 của corpus | Build bằng CPU nếu thiếu |
| `parent_embeddings.pkl` | Embedding parent chunk | Build bằng H100 nếu thiếu |
| `child_embeddings.pkl` | Embedding child chunk | Build bằng H100 nếu thiếu |

Pipeline kiểm tra các mốc cấu trúc sau để phát hiện dữ liệu hoặc cache không đồng nhất:

| Thành phần | Số lượng |
|---|---:|
| Tài liệu corpus | 8.532 |
| Tài liệu có parent chunk | 8.512 |
| Parent chunk | 738.506 |
| Tài liệu có child chunk | 7.078 |
| Child chunk | 595.597 |

Có 20 tài liệu không tạo ra parent chunk sau bước tiền xử lý; vì vậy 8.512 parent document là trạng thái hợp lệ.

Bốn cache phục vụ tạo đặc trưng và huấn luyện:

```text
top_documents.pkl
candidate_audit.pkl
pair_features.pkl
singleton_features.pkl
```

Các file `.pkl` lớn được loại khỏi Git bằng `.gitignore`. Chúng được giữ trên máy chạy hoặc Modal Volume, không cần đưa lên GitHub.

## 5. Chạy bằng Modal H100

Modal phù hợp khi máy cá nhân không có đủ GPU hoặc RAM. Workflow tự kiểm tra dữ liệu và cache, chỉ build phần còn thiếu, phân bổ đúng loại tài nguyên cho từng công việc và tải các output chính về máy.

### 5.1. Clone repository

```powershell
git clone https://github.com/Vlus06/DSC_TASK2.git
cd DSC_TASK2
```

### 5.2. Cài Modal CLI

```powershell
py -3.12 -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
python -m pip install modal
```

Nếu PowerShell chặn `Activate.ps1`, có thể gọi trực tiếp `.\.venv\Scripts\python.exe` như các lệnh bên dưới.

### 5.3. Kết nối tài khoản Modal

```powershell
.\.venv\Scripts\python.exe -m modal setup
```

Modal CLI sẽ mở trình duyệt để xác thực. Không lưu token vào source code, notebook hoặc Git.

### 5.4. Smoke test

Chạy thử 10 câu để kiểm tra môi trường, model và định dạng output:

```powershell
.\.venv\Scripts\python.exe -m modal run .\modal_app.py `
  --public-limit 10 `
  --ce-batch-size 32
```

Smoke test tạo file `submission_smoke_10.json`. File này chỉ dùng để kiểm tra pipeline.

### 5.5. Chạy đầy đủ tập public

```powershell
.\.venv\Scripts\python.exe -m modal run .\modal_app.py `
  --public-limit 0 `
  --ce-batch-size 32 `
  --checkpoint-every 25
```

| Tham số | Mặc định | Ý nghĩa |
|---|---:|---|
| `--public-limit` | `0` | `0` chạy toàn bộ tập public; số dương dùng để smoke test |
| `--ce-batch-size` | `32` | Batch size của cross-encoder trên H100 |
| `--checkpoint-every` | `25` | Chu kỳ lưu tiến độ theo số câu hỏi |

Mỗi lần chạy sinh một `run_id`. ID được hiển thị trong terminal:

```text
Starting cost-aware workflow; Run ID: <run_id>
```

Giữ lại `run_id` để tra log hoặc tải output từ Modal Volume.

### 5.6. Chạy tách khỏi terminal

```powershell
.\.venv\Scripts\python.exe -m modal run --detach .\modal_app.py `
  --public-limit 0 `
  --ce-batch-size 32 `
  --checkpoint-every 25
```

Có thể đóng terminal sau khi app đã khởi động. Trạng thái và log được theo dõi trên Modal Dashboard.

## 6. Luồng xử lý tự động trên Modal

Một lệnh `modal run` thực hiện lần lượt:

1. kiểm tra dataset, stopwords và cache trên các Modal Volume;
2. upload những input có ở local nhưng còn thiếu trên Volume;
3. đóng gói corpus thành một file để tránh đọc lại 8.532 JSON nhỏ;
4. build BM25 bằng CPU nếu chưa có;
5. build dense parent bằng H100 nếu chưa có;
6. build child embedding bằng H100 nếu chưa có;
7. build hoặc đọc lại bốn cache đặc trưng;
8. tập hợp đặc trưng xếp hạng và huấn luyện ensemble năm XGBRanker bằng CPU;
9. inference tập public bằng H100;
10. lưu model, log, manifest và file dự đoán;
11. tải các output chính về máy nếu terminal vẫn kết nối.

Phân bổ tài nguyên:

| Giai đoạn | Tài nguyên Modal | Công việc |
|---|---|---|
| Kiểm tra input/cache | 2 CPU, 8 GiB RAM | Đọc trạng thái Volume |
| Đóng gói corpus | 8 CPU, 16 GiB RAM | Gom passage, name và link |
| Build BM25 | 16 CPU, 32 GiB RAM | Tokenize và tạo index |
| Build parent/child | H100, 16 CPU, 128 GiB RAM | Chia chunk và sinh embedding |
| Build cache đặc trưng | H100, 16 CPU, 128 GiB RAM | Retrieval, query embedding và reranker |
| Train ensemble XGBRanker | 32 CPU, 64 GiB RAM | Assemble feature và huấn luyện năm ranker |
| Inference public | H100, 16 CPU, 128 GiB RAM | Dense retrieval và cross-encoder |

Parent và child cache lưu embedding NumPy `float32`. Cache tạo bằng T4 có thể đọc và sử dụng trên H100. Builder chọn CUDA theo môi trường, không khóa theo tên GPU và tắt TF32 khi tạo embedding.

## 7. Cơ chế sử dụng lại cache

Workflow xử lý từng artifact độc lập:

- cache đã có trên Volume: bỏ qua bước build tương ứng;
- cache chưa có trên Volume nhưng có ở local: upload một lần;
- cache không có ở cả hai nơi: tự build bằng CPU hoặc H100;
- cache đặc trưng đang làm dở: tiếp tục các qid còn thiếu;
- cache đặc trưng đã hoàn tất: bỏ qua toàn bộ bước chuẩn bị;
- model XGBoost: huấn luyện bằng CPU từ tập train;
- inference: lưu progress định kỳ trong thư mục của run hiện tại.

Modal sử dụng ba Volume:

| Volume | Nội dung |
|---|---|
| `legalqa-data` | Dataset, stopwords và toàn bộ cache |
| `legalqa-models` | Model XGBoost và model tải từ Hugging Face |
| `legalqa-results` | Output theo từng `run_id` |

Không xóa Volume nếu muốn dùng lại cache và model đã tải trong lần chạy sau.

## 8. Tệp đầu ra

Khi terminal duy trì kết nối đến cuối, các file chính được tải về:

```text
outputs/modal/<run_id>/
```

Nội dung thư mục:

```text
submission.json
inference_log.json
run_manifest.json
training_manifest.json
01_prepare_features.log
02_train_rankers.log
03_infer_public.log
```

File dự đoán cho toàn bộ tập public:

```text
submission.json
```

Output đồng thời được lưu trên Modal tại:

```text
legalqa-results/<run_id>/
```

Nếu chạy với `--detach` hoặc terminal bị đóng trước bước tải, dùng:

```powershell
.\.venv\Scripts\python.exe -m modal volume get legalqa-results /<run_id> .\outputs\modal\<run_id>
```

Thay `<run_id>` bằng ID được in khi workflow bắt đầu.

Các dòng log xác nhận pipeline hoàn tất:

```text
[train] five rankers saved
[infer] DONE
[workflow] DONE: /results/<run_id>
Downloaded outputs to: ...\outputs\modal\<run_id>
```

## 9. Chạy trên máy local

Máy local cần đủ RAM để nạp các cache embedding dung lượng lớn và cần ba base cache trong `data/cache/`.

Cài thư viện:

```powershell
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
```

Kiểm tra dataset và base cache:

```powershell
.\.venv\Scripts\python.exe -u scripts\run.py --mode check
```

Chạy toàn bộ pipeline:

```powershell
.\.venv\Scripts\python.exe -u scripts\run.py `
  --mode infer `
  --device cuda `
  --ce-batch-size 32 `
  --checkpoint-every 25
```

Các mode hỗ trợ:

| Mode | Chức năng |
|---|---|
| `check` | Kiểm tra dataset và cấu trúc ba base cache |
| `prepare` | Tạo hoặc đọc bốn cache đặc trưng rồi dừng |
| `infer` | Chuẩn bị cache còn thiếu, huấn luyện năm ranker và inference public |

Giới hạn số câu public để kiểm tra nhanh:

```powershell
.\.venv\Scripts\python.exe -u scripts\run.py `
  --mode infer `
  --device cuda `
  --public-limit 10
```

Trên Linux, dùng `python` thay cho đường dẫn Python của `.venv` trên Windows và dùng `\` để tiếp dòng.

## 10. Theo dõi và dừng tiến trình

Terminal chạy `modal run` stream log theo thời gian thực. Modal Dashboard hiển thị app, function, container, thời gian chạy và chi phí sử dụng.

Để dừng:

- tiến trình đang gắn với terminal: nhấn `Ctrl+C`;
- app chạy detached: mở Modal Dashboard và chọn **Stop app**.

Các cache đã commit trên Volume vẫn được giữ lại sau khi app dừng.

## 11. Kiểm tra file dự đoán

Trước khi sử dụng file đầu ra, kiểm tra:

1. đây là file `submission.json` của lần chạy đầy đủ;
2. inference log có `errors = 0`;
3. file chứa đủ qid của tập public;
4. không có câu trả lời rỗng;
5. không nhầm với file `submission_smoke_<n>.json`.

## 12. Xử lý lỗi thường gặp

### Modal báo chưa đăng nhập

```powershell
.\.venv\Scripts\python.exe -m modal setup
```

### Log chưa cập nhật trong lúc load cache

Dense parent và child cache có kích thước vài GB. Việc đọc và giải tuần tự pickle có thể mất thời gian mà chưa in thêm log. Nếu container vẫn ở trạng thái `Running` và không có traceback, tiến trình vẫn đang hoạt động.

### App hiện `Finished` và có error

Mở tab **Logs**, chọn đúng app và đọc traceback cuối cùng. Log của từng stage cũng được lưu trong `legalqa-results/<run_id>/` nếu thư mục output đã được tạo.

### Terminal đóng trước khi tải file

Output vẫn nằm trên `legalqa-results`. Dùng lệnh `modal volume get` tại mục 8.

### Hết bộ nhớ GPU

Giảm `--ce-batch-size`, ví dụ từ `32` xuống `16`.

### GitHub không nhận cache lớn

Đây là hành vi dự kiến. `data/cache/*.pkl` được `.gitignore` loại trừ. Giữ cache trên máy hoặc Volume `legalqa-data`; workflow sẽ upload file local còn thiếu hoặc tự build trên Modal.

## 13. Các file mã nguồn quan trọng

- `scripts/run.py`: entrypoint local.
- `scripts/pipeline.py`: điều phối các chế độ chạy.
- `scripts/build_bm25.py`: tạo BM25 index.
- `scripts/build_dense_parent.py`: chia parent chunk và tạo embedding.
- `scripts/build_child.py`: chia child chunk và tạo embedding.
- `scripts/modal_stages.py`: triển khai từng stage trên Modal.
- `src/legalqa/engine.py`: retrieval, feature, ranker và answer builder.
- `src/legalqa/artifacts.py`: quản lý cache đặc trưng và checkpoint.
- `src/legalqa/settings.py`: cấu hình pipeline.
- `modal_app.py`: phân bổ CPU/H100, Volume và luồng chạy cloud.
