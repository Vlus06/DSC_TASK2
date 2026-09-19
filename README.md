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
│   │   ├── private-official.json
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
data/TASK2/private-official.json    # chỉ bắt buộc khi chạy private test
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

### 5.4. Chuẩn bị dataset trên máy

Sau khi clone repository, đặt dữ liệu đúng cấu trúc sau:

```text
data/
├── TASK2/
│   ├── train.json
│   ├── public-official.json
│   ├── private-official.json
│   └── selected-contexts/
│       ├── context_....json
│       └── ...
├── cache/
└── stopwords.txt
```

`selected-contexts/` phải chứa đủ các file JSON của corpus. Với bộ dữ liệu dùng trong project, code kiểm tra đúng 8.532 tài liệu. `private-official.json` chỉ cần khi chạy private test. Không đổi tên `train.json`, `public-official.json`, `private-official.json`, `selected-contexts` hoặc `stopwords.txt` vì workflow dùng các đường dẫn này để nhận diện dữ liệu.

Đặt biến mã hóa UTF-8 trước khi dùng Modal CLI trên Windows để terminal in được log tiếng Việt và ký hiệu kiểm tra:

```powershell
$env:PYTHONIOENCODING = "utf-8"
```

### 5.5. Nạp dataset lên Modal Volume

Workflow sử dụng Volume `legalqa-data`. Tạo Volume một lần; nếu Volume đã tồn tại thì bỏ qua lệnh tạo:

```powershell
.\.venv\Scripts\python.exe -m modal volume create legalqa-data
```

Nạp dataset và stopwords:

```powershell
.\.venv\Scripts\python.exe -m modal volume put --force `
  legalqa-data .\data\TASK2 /TASK2

.\.venv\Scripts\python.exe -m modal volume put --force `
  legalqa-data .\data\stopwords.txt /stopwords.txt
```

Lần chạy `modal_app.py` đầu tiên cũng có thể tự upload hai đầu vào này nếu chúng còn thiếu trên Volume. Các lệnh `volume put` ở trên hữu ích khi muốn chuẩn bị và kiểm tra dữ liệu trước khi bắt đầu pipeline dài.

Kiểm tra các file đã có trên Volume:

```powershell
.\.venv\Scripts\python.exe -m modal volume ls legalqa-data /TASK2
.\.venv\Scripts\python.exe -m modal volume ls legalqa-data /cache
```

### 5.6. Tạo và sử dụng `corpus_metadata.pkl`

`corpus_metadata.pkl` là bản đóng gói của 8.532 file trong `selected-contexts/`. File chứa ba bảng ánh xạ `passages`, `names` và `links`. Nó giúp các stage sau đọc corpus từ một file pickle thay vì mở lại hàng nghìn file JSON nhỏ.

Sau khi dataset đã nằm trên `legalqa-data`, tạo file này bằng CPU trên Modal:

```powershell
.\.venv\Scripts\python.exe -m modal run .\modal_app.py::pack_corpus
```

Code thực hiện nằm trong hàm `pack_corpus()` của `scripts/modal_stages.py`. Stage đọc toàn bộ `data/TASK2/selected-contexts/*.json`, kiểm tra đủ 8.532 tài liệu rồi lưu file bền vững tại:

```text
Volume: legalqa-data
Đường dẫn: /cache/corpus_metadata.pkl
```

Nếu đã có bản backup `data/cache/corpus_metadata.pkl` trên máy, có thể nạp thẳng lên Volume và bỏ qua bước quét 8.532 file JSON:

```powershell
.\.venv\Scripts\python.exe -m modal volume put --force `
  legalqa-data .\data\cache\corpus_metadata.pkl /cache/corpus_metadata.pkl
```

Có thể tải file đã build trên Modal về máy để dùng lại cho lần sau:

```powershell
.\.venv\Scripts\python.exe -m modal volume get --force `
  legalqa-data /cache/corpus_metadata.pkl .\data\cache
```

Không cần chạy lại `pack_corpus` nếu `/cache/corpus_metadata.pkl` đã tồn tại và có dữ liệu. Workflow đầy đủ cũng tự gọi stage này khi file còn thiếu, nhưng chạy riêng trước giúp xác nhận corpus đã được đóng gói trước khi bắt đầu các bước tốn thời gian hơn.

Kiểm tra toàn bộ trạng thái đầu vào và cache:

```powershell
.\.venv\Scripts\python.exe -m modal run .\modal_app.py::inspect_inputs
```

Trạng thái sẵn sàng tối thiểu trước lần chạy đầu phải có:

```text
"dataset_ready": true
"stopwords_ready": true
"packed_corpus": true
```

Các mục BM25, parent, child, feature cache và model có thể là `false`; workflow sẽ build phần còn thiếu rồi lưu lại trên Volume.

### 5.7. Tải cache dựng sẵn từ Google Drive

Để không phải build lại corpus, BM25, parent embedding, child embedding và bốn cache đặc trưng, có thể tải bộ cache dựng sẵn tại [Google Drive — DSC_TASK2_2026](https://drive.google.com/drive/folders/137SYXPgpX82kn1-DZIrFGeZwQkk72cDY).

Sau khi tải, đặt toàn bộ file vào `data/cache/` và kiểm tra đúng tên:

```text
data/cache/
├── bm25_index.pkl
├── parent_embeddings.pkl
├── child_embeddings.pkl
├── corpus_metadata.pkl
├── top_documents.pkl
├── candidate_audit.pkl
├── pair_features.pkl
└── singleton_features.pkl
```

Tên file phải khớp hoàn toàn với danh sách trên

Nạp toàn bộ thư mục cache lên Modal Volume:

```powershell
.\.venv\Scripts\python.exe -m modal volume put --force `
  legalqa-data .\data\cache /cache
```

Sau khi upload, kiểm tra:

```powershell
.\.venv\Scripts\python.exe -m modal run .\modal_app.py::inspect_inputs
```

Kết quả phải có `packed_corpus: true`; ba mục trong `base` và bốn mục trong `feature_caches` đều là `true`. Bộ cache trên Drive không thay thế dataset: vẫn phải nạp `train.json`, `public-official.json`, thư mục `selected-contexts/` và `stopwords.txt` theo mục 5.5.

Nếu `feature_marker` là `false` trong lần nhập cache đầu tiên, workflow sẽ đọc và kiểm tra bốn cache đặc trưng rồi tạo `feature_cache_complete.json` trên Volume. Các cache corpus, BM25 và embedding hợp lệ vẫn được dùng lại, không bị build lại.

### 5.8. Smoke test

Chạy thử 10 câu để kiểm tra môi trường, model và định dạng output:

```powershell
.\.venv\Scripts\python.exe -m modal run .\modal_app.py `
  --public-limit 10 `
  --ce-batch-size 32
```

Smoke test tạo file `submission_smoke_10.json`. File này chỉ dùng để kiểm tra pipeline.

### 5.9. Chạy đầy đủ tập public

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

### 5.10. Chạy tách khỏi terminal

```powershell
.\.venv\Scripts\python.exe -m modal run --detach .\modal_app.py `
  --public-limit 0 `
  --ce-batch-size 32 `
  --checkpoint-every 25
```

Có thể đóng terminal sau khi app đã khởi động. Trạng thái và log được theo dõi trên Modal Dashboard.

### 5.11. Chạy private test và tạo file nộp

Đặt tập private đúng tên và đúng vị trí trên máy:

```text
data/TASK2/private-official.json
```

Kiểm tra trạng thái dataset, cache và model trên Modal:

```powershell
.\.venv\Scripts\python.exe -m modal run .\modal_app.py::inspect_inputs
```

Nếu file đã có trên Volume, kết quả sẽ hiển thị `private_ready: true`. Nếu `private_ready` còn là `false` nhưng file private đã có trên máy, lệnh chạy private bên dưới sẽ tự upload file lên `legalqa-data/TASK2/private-official.json` rồi kiểm tra lại.

Có thể smoke test 10 câu trước:

```powershell
.\.venv\Scripts\python.exe -m modal run .\modal_app.py `
  --dataset private `
  --private-limit 10 `
  --ce-batch-size 32 `
  --checkpoint-every 25
```

Smoke test tạo `submission_private_smoke_10.json`; không dùng file này để nộp.

Chạy toàn bộ private test:

```powershell
.\.venv\Scripts\python.exe -m modal run .\modal_app.py `
  --dataset private `
  --private-limit 0 `
  --ce-batch-size 32 `
  --checkpoint-every 25
```

`--private-limit 0` có nghĩa là chạy toàn bộ tập private. Workflow tái sử dụng corpus đã đóng gói, BM25, embedding parent/child và năm ranker đã có; nhánh này không build cache, prepare feature hoặc train lại model. Inference chạy trên H100 và lưu checkpoint sau mỗi 25 câu.

Khi hoàn tất, terminal in:

```text
Starting private inference; Run ID: private-<id>
Downloaded private outputs to: ...\outputs\modal\private-<id>
```

File dùng để nộp là:

```text
outputs/modal/private-<id>/submission_private.json
```

Tìm nhanh file private mới nhất:

```powershell
Get-ChildItem .\outputs\modal -Recurse -Filter submission_private.json |
  Sort-Object LastWriteTime -Descending |
  Select-Object -First 1 FullName, Length, LastWriteTime
```

Nếu terminal bị ngắt hoặc một số câu bị lỗi, giữ lại `run_id` và chạy tiếp sau khi đã sửa nguyên nhân:

```powershell
.\.venv\Scripts\python.exe -m modal run .\modal_app.py `
  --dataset private `
  --private-limit 0 `
  --resume-run-id private-<id> `
  --ce-batch-size 32 `
  --checkpoint-every 25
```

Chế độ resume đọc checkpoint trong chính thư mục run cũ, giữ các câu đã thành công và chỉ chạy lại câu bị lỗi hoặc có answer rỗng.

Nếu inference đã xong trên Modal nhưng terminal không tải file về, dùng chính `run_id` đó:

```powershell
.\.venv\Scripts\python.exe -m modal volume get --force `
  legalqa-results "/private-<id>" ".\outputs\modal\private-<id>"
```

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

Luồng trên áp dụng cho public inference hoặc lần thiết lập ban đầu. Khi chọn `--dataset private`, workflow xác minh dataset, ba base cache và năm ranker đã sẵn sàng, sau đó chạy thẳng private inference trên H100.

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
| Inference private | H100, 16 CPU, 128 GiB RAM | Tái sử dụng cache/model và sinh file nộp private |

Parent và child cache lưu embedding NumPy `float32`. Cache tạo bằng T4 có thể đọc và sử dụng trên H100. Builder chọn CUDA theo môi trường, không khóa theo tên GPU và tắt TF32 khi tạo embedding.

## 7. Cơ chế sử dụng lại cache

Workflow xử lý từng artifact độc lập:

- cache đã có trên Volume: bỏ qua bước build tương ứng;
- cache chưa có trên Volume nhưng có ở local: upload một lần;
- cache không có ở cả hai nơi: tự build bằng CPU hoặc H100;
- cache đặc trưng đang làm dở: tiếp tục các qid còn thiếu;
- cache đặc trưng đã hoàn tất: bỏ qua toàn bộ bước chuẩn bị;
- đủ năm model XGBoost và manifest hợp lệ: bỏ qua bước train;
- thiếu model hoặc cache đặc trưng vừa được build lại: huấn luyện lại bằng CPU;
- inference: lưu progress định kỳ trong thư mục của run hiện tại.

Modal sử dụng ba Volume:

| Volume | Nội dung |
|---|---|
| `legalqa-data` | Dataset, stopwords và toàn bộ cache |
| `legalqa-models` | Model XGBoost và model tải từ Hugging Face |
| `legalqa-results` | Output theo từng `run_id` |

Không xóa Volume nếu muốn dùng lại cache và model đã tải trong lần chạy sau.

Năm model XGBoost được lưu bền vững tại:

```text
legalqa-models/legalqa/
├── answer_ranker_seed_42.json
├── answer_ranker_seed_10042.json
├── answer_ranker_seed_20042.json
├── answer_ranker_seed_30042.json
├── answer_ranker_seed_40042.json
└── training_manifest.json
```

Tải toàn bộ model về máy để backup hoặc tái sử dụng:

```powershell
.\.venv\Scripts\python.exe -m modal volume get --force legalqa-models /legalqa .\data\models
```

Sau khi tải, các file nằm trong `data/models/legalqa/`.

Thư mục `data/models/` được loại khỏi Git vì đây là artifact sinh ra sau khi train.

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

Với private inference, thư mục output có dạng:

```text
outputs/modal/private-<id>/
├── submission_private.json
├── inference_log_private.json
├── run_manifest.json
└── 03_infer_private.log
```

`submission_private.json` là file dùng để nộp private test. `inference_log_private.json` chứa số câu đã xử lý và danh sách lỗi nếu có.

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

Với private test, chọn đúng `submission_private.json`, không chọn `submission_private_smoke_<n>.json`. Kiểm tra nhanh file private mới nhất bằng PowerShell:

```powershell
$file = Get-ChildItem .\outputs\modal -Recurse -Filter submission_private.json |
  Sort-Object LastWriteTime -Descending |
  Select-Object -First 1

$json = Get-Content -Raw $file.FullName | ConvertFrom-Json
$rows = @($json.PSObject.Properties)
$empty = @($rows | Where-Object {
  [string]::IsNullOrWhiteSpace([string]$_.Value.answer)
})
$logFile = Join-Path $file.DirectoryName "inference_log_private.json"
$log = Get-Content -Raw $logFile | ConvertFrom-Json
$errors = @($log | Where-Object { $null -ne $_.error })

[pscustomobject]@{
  File = $file.FullName
  Questions = $rows.Count
  EmptyAnswers = $empty.Count
  Errors = $errors.Count
}
```

Với bộ private hiện tại, kết quả hợp lệ phải có `Questions = 1918`, `EmptyAnswers = 0` và `Errors = 0`.

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

### Private inference dừng sau khi đã chạy nhiều câu

Không tạo `run_id` mới. Ghi lại `private-<id>` của lần chạy bị dừng, sửa nguyên nhân trong code nếu có traceback, sau đó chạy lại với `--resume-run-id private-<id>` theo mục 5.11. Pipeline sẽ tái sử dụng checkpoint và chỉ xử lý lại câu lỗi hoặc answer rỗng.

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
