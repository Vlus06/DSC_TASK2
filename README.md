# LegalQA Task 2 — Hệ thống hỏi đáp pháp luật tiếng Việt

Repository triển khai pipeline truy hồi, chọn bằng chứng và sinh câu trả lời cho bộ dữ liệu LegalQA Task 2. Luồng production dùng bộ chọn V2.1 đã đóng băng; các công thức retrieval, ACTION40, answer builder và `postprocess_best_06252` được giữ nguyên.

## Kiến trúc

```text
question
  → BM25 top 400 + dense document top 250
  → weighted fusion top 30 → top 3 documents
  → parent/child candidate pool
  → dense top 10 chunks
  → AITeamVN/Vietnamese_Reranker top 5
  → 5 singleton + 10 unordered pairs = 15 actions
  → ACTION40
  → 5 retarget XGBRanker + 10 heterogeneous level-0 models
  → xgb_rank_meta
  → safety mask theo primary_doc + first_article
  → pairwise_regret_v2 + confidence gate 0.55
  → selected action
  → answer builder, tối đa 5.800 ký tự
  → postprocess_best_06252
  → submission.json
```

Bi-encoder là `AITeamVN/Vietnamese_Embedding`; reranker là `AITeamVN/Vietnamese_Reranker`. Bộ chọn cuối chỉ học trên `scale5000`, được tạo bằng `random.Random(42).shuffle(qids)` rồi lấy vị trí `2000:7000`. `base1000`, `fresh1000` và nhãn private không được dùng để train bộ chọn production.

## Cấu trúc chính

```text
src/legalqa/
├── engine.py             # retrieval, ACTION40, answer builder, public predict
├── selector_models.py    # 5 retarget + 10 model level-0
├── meta_selector.py      # transforms, safety, meta, pairwise, gate 0.55
├── selector_training.py  # final targets, OOF 5 fold, resume, full training
├── model_bundle.py       # native model files và manifest validation
├── postprocess_v87.py    # postprocess_best_06252 hiện hành
└── artifacts.py          # dataset và feature caches
scripts/
├── run.py                # entrypoint local
├── pipeline.py           # check/prepare/train/infer
└── modal_stages.py       # stage độc lập trên Modal
modal_app.py              # phân bổ CPU/H100 và Modal Volumes
```

## Dữ liệu và cache

Đặt dữ liệu theo cấu trúc:

```text
data/
├── TASK2/
│   ├── train.json
│   ├── public-official.json
│   ├── private-official.json
│   └── selected-contexts/
├── cache/
│   ├── corpus_metadata.pkl
│   ├── bm25_index.pkl
│   ├── parent_embeddings.pkl
│   ├── child_embeddings.pkl
│   ├── top_documents.pkl
│   ├── candidate_audit.pkl
│   ├── pair_features.pkl
│   └── singleton_features.pkl
└── stopwords.txt
```

`corpus_metadata.pkl` đóng gói passage, tên và đường dẫn của 8.532 văn bản, giúp Modal không phải mở lại hàng nghìn JSON nhỏ. Các file cache dựng sẵn có thể tải tại [Google Drive — DSC_TASK2_2026](https://drive.google.com/drive/folders/137SYXPgpX82kn1-DZIrFGeZwQkk72cDY). Cache phải đúng bộ dữ liệu và đúng tên; training còn kiểm tra split, phiên bản và chữ ký protocol trước khi tái sử dụng.

Các cache và model lớn không đưa lên GitHub. Chúng được lưu local hoặc trên Modal Volume.

## Cài đặt local

```powershell
py -3.12 -m venv .venv
.\.venv\Scripts\python.exe -m pip install --upgrade pip
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
```

Kiểm tra dataset và ba base cache:

```powershell
.\.venv\Scripts\python.exe -u scripts\run.py --mode check
```

Tạo hoặc tiếp tục bốn feature cache:

```powershell
.\.venv\Scripts\python.exe -u scripts\run.py `
  --mode prepare `
  --device cuda `
  --ce-batch-size 32 `
  --checkpoint-every 25
```

Train bundle V2.1 trên CPU. Lệnh này không tải embedding/reranker và không dùng GPU:

```powershell
.\.venv\Scripts\python.exe -u scripts\run.py `
  --mode train-selector-v21 `
  --model-dir .\data\models\legalqa_selector_v21 `
  --checkpoint-every 25
```

Training tạo `75.000` final targets, prediction OOF 5 fold cho `5.000 × 15 = 75.000` rows, hai model level-2 và 15 model level-0 full-scale. Các bước đắt được lưu tại:

```text
data/cache/selector_v21_final_targets.pkl
data/cache/selector_v21_oof_fold_1.pkl
...
data/cache/selector_v21_oof_fold_5.pkl
```

Nếu cache đã hoàn chỉnh và chữ ký khớp, lần chạy sau tự dùng lại. Nếu bundle hoàn chỉnh tồn tại và vượt qua manifest/hash validation, training được bỏ qua.

Inference public:

```powershell
.\.venv\Scripts\python.exe -u scripts\run.py `
  --mode infer `
  --split public `
  --model-dir .\data\models\legalqa_selector_v21 `
  --device cuda `
  --public-limit 0
```

Inference private:

```powershell
.\.venv\Scripts\python.exe -u scripts\run.py `
  --mode infer `
  --split private `
  --model-dir .\data\models\legalqa_selector_v21 `
  --device cuda `
  --private-limit 0
```

Inference không train model. Nó từ chối bundle cũ hoặc manifest không đúng protocol V2.1. Progress được checkpoint riêng cho public/private và chỉ dùng lại khi đúng phiên bản selector.

## Bundle model

Bundle mặc định nằm tại `data/models/legalqa_selector_v21/` và gồm đúng 17 native model files:

- 5 `retarget_<seed>.json`;
- 10 `heterogeneous_<name>.*`;
- `xgb_rank_meta.json`;
- `pairwise_regret_v2.json`;
- `selector_manifest.json`.

Manifest bắt buộc xác nhận:

```text
version: legalqa_selector_v21_gate055_06299_v1
train_split: scale5000_seed42_positions_2000_6999
action_feature_count: 40
meta_feature_count: 36
v21_feature_count: 109
training_qid_count: 5000
final_target_rows: 75000
oof_rows: 75000
model_count: 17
gate_threshold: 0.55
postprocess: best_06252
```

Mỗi model có kích thước và SHA256 trong manifest. Inference kiểm tra schema và hash trước khi nạp.

## Chạy trên Modal

Có hai cách chuẩn bị pipeline:

1. **Dùng cache dựng sẵn từ Google Drive:** nhanh nhất, bỏ qua các bước quét corpus, BM25 và sinh embedding parent/child.
2. **Chạy lại từ đầu:** bắt đầu từ dataset gốc, lần lượt tạo toàn bộ cache, train selector rồi inference private.

Cả hai cách đều kết thúc bằng `submission_private.zip`. File ZIP dùng để nộp chỉ chứa một file `submission.json`.

### Thiết lập chung

Clone hoặc cập nhật repository:

```powershell
git clone https://github.com/Vlus06/DSC_TASK2.git
cd DSC_TASK2
```

Nếu đã clone trước đó:

```powershell
git pull origin main
```

Tạo môi trường và cài Modal CLI:

```powershell
py -3.12 -m venv .venv
.\.venv\Scripts\python.exe -m pip install --upgrade pip
.\.venv\Scripts\python.exe -m pip install modal
.\.venv\Scripts\python.exe -m modal setup
$env:PYTHONIOENCODING = "utf-8"
```

Tạo ba Volume. Nếu Volume đã tồn tại, Modal sẽ báo và có thể tiếp tục:

```powershell
.\.venv\Scripts\python.exe -m modal volume create legalqa-data
.\.venv\Scripts\python.exe -m modal volume create legalqa-models
.\.venv\Scripts\python.exe -m modal volume create legalqa-results
```

Vai trò của từng Volume:

| Volume | Nội dung |
|---|---|
| `legalqa-data` | Dataset, stopwords, base cache, feature cache, final-target cache và OOF cache |
| `legalqa-models` | Hugging Face cache và bundle selector V2.1 gồm 17 model |
| `legalqa-results` | Log, checkpoint inference và submission theo `run_id` |

### Cách 1 — Dùng cache từ Google Drive

#### Bước 1: tải cache

Tải các file trong [Google Drive — DSC_TASK2_2026](https://drive.google.com/drive/folders/137SYXPgpX82kn1-DZIrFGeZwQkk72cDY), rồi đặt vào `data/cache/`.

Kiểm tra đúng tám tên file sau:

```text
data/cache/
├── corpus_metadata.pkl
├── bm25_index.pkl
├── parent_embeddings.pkl
├── child_embeddings.pkl
├── top_documents.pkl
├── candidate_audit.pkl
├── pair_features.pkl
└── singleton_features.pkl
```

Nếu file tải về có tên `parent_embeddings` hoặc `child_embeddings` nhưng không có đuôi, hãy bật **File name extensions** trong Windows Explorer và đổi thành `parent_embeddings.pkl` hoặc `child_embeddings.pkl`. Không đổi các tên khác.

Bộ cache phải được dùng cùng dataset đã tạo ra nó. Training V2.1 kiểm tra version, qid, action schema, candidate text, score, raw target và SHA256 protocol; cache không tương thích sẽ bị từ chối thay vì âm thầm dùng tiếp.

#### Bước 2: chuẩn bị dataset local

Đặt dữ liệu theo cấu trúc:

```text
data/
├── TASK2/
│   ├── train.json
│   ├── public-official.json
│   ├── private-official.json
│   └── selected-contexts/
│       └── selected-contexts/
│           ├── context_....json
│           └── ...
├── cache/
│   └── tám file cache ở bước 1
└── stopwords.txt
```

`private-official.json` được `.gitignore` loại trừ và không được đưa lên GitHub.

#### Bước 3: upload dataset, stopwords và cache

```powershell
.\.venv\Scripts\python.exe -m modal volume put --force `
  legalqa-data .\data\TASK2 /TASK2

.\.venv\Scripts\python.exe -m modal volume put --force `
  legalqa-data .\data\stopwords.txt /stopwords.txt

.\.venv\Scripts\python.exe -m modal volume put --force `
  legalqa-data .\data\cache /cache
```

#### Bước 4: kiểm tra cache trên Modal

```powershell
.\.venv\Scripts\python.exe -m modal run .\modal_app.py::inspect_inputs
```

Trước khi train, cần thấy:

```text
dataset_ready: true
private_ready: true
stopwords_ready: true
packed_corpus: true
base.bm25: true
base.dense: true
base.child: true
feature_caches.top_documents: true
feature_caches.candidate_audit: true
feature_caches.pair_features: true
feature_caches.singleton_features: true
```

`selector_ready: false` là trạng thái bình thường nếu chưa train bundle V2.1. `feature_marker` không bắt buộc đối với lệnh train trực tiếp; nội dung cache vẫn được kiểm tra trước khi sử dụng.

Nếu một cache hiển thị `false`, kiểm tra tên và vị trí:

```powershell
.\.venv\Scripts\python.exe -m modal volume ls legalqa-data /cache
```

#### Bước 5: train selector V2.1

```powershell
.\.venv\Scripts\python.exe -m modal run .\modal_app.py::train_selector_v21 `
  --run-id selector-v21-train
```

Stage CPU này thực hiện:

1. tái lập raw target trên mẫu cache lịch sử;
2. tạo `5.000 × 15 = 75.000` final targets sau `postprocess_best_06252`;
3. train và dự đoán 5 fold OOF, mỗi fold train trên 4.000 qid và dự đoán 1.000 qid;
4. ghép đủ 75.000 OOF rows;
5. train `xgb_rank_meta` và `pairwise_regret_v2`;
6. train 5 retarget + 10 heterogeneous models trên toàn bộ `scale5000`;
7. lưu bundle 17 model cùng manifest và SHA256.

Target cache và mỗi fold OOF được lưu trên `legalqa-data:/cache`. Nếu stage bị dừng, chạy lại đúng lệnh trên; phần hoàn chỉnh có chữ ký khớp sẽ được dùng lại.

Theo dõi log trực tiếp trong terminal hoặc Modal Dashboard. Không bắt đầu private inference cho tới khi stage train kết thúc.

#### Bước 6: xác nhận bundle

```powershell
.\.venv\Scripts\python.exe -m modal run .\modal_app.py::inspect_inputs
```

Cần thấy:

```text
selector_manifest: true
selector_ready: true
```

Toàn bộ 17 mục trong `selector_files` phải là `true`.

#### Bước 7: smoke test private

```powershell
.\.venv\Scripts\python.exe -m modal run .\modal_app.py `
  --dataset private `
  --private-limit 10 `
  --ce-batch-size 32 `
  --checkpoint-every 5
```

Smoke test tạo `submission_private_smoke_10.zip`; file này chỉ dùng để kiểm tra môi trường.

#### Bước 8: chạy toàn bộ private test

```powershell
.\.venv\Scripts\python.exe -m modal run .\modal_app.py `
  --dataset private `
  --private-limit 0 `
  --ce-batch-size 32 `
  --checkpoint-every 25
```

`--private-limit 0` chạy toàn bộ tập private. Terminal sẽ in `Run ID` dạng `private-<id>` và tự tải output về:

```text
outputs/modal/private-<id>/submission_private.zip
```

### Cách 2 — Chạy lại hoàn toàn từ đầu

Cách này chỉ cần dataset gốc và `stopwords.txt`. Không đặt các file `.pkl` tải từ Drive vào `data/cache/`.

#### Bước 1: chuẩn bị và upload dữ liệu gốc

Chuẩn bị:

```text
data/TASK2/train.json
data/TASK2/public-official.json
data/TASK2/private-official.json
data/TASK2/selected-contexts/selected-contexts/*.json
data/stopwords.txt
```

Upload:

```powershell
.\.venv\Scripts\python.exe -m modal volume put --force `
  legalqa-data .\data\TASK2 /TASK2

.\.venv\Scripts\python.exe -m modal volume put --force `
  legalqa-data .\data\stopwords.txt /stopwords.txt
```

Kiểm tra:

```powershell
.\.venv\Scripts\python.exe -m modal run .\modal_app.py::inspect_inputs
```

Lúc này `dataset_ready`, `private_ready` và `stopwords_ready` phải là `true`; các cache có thể là `false`.

#### Bước 2: đóng gói corpus bằng CPU

```powershell
.\.venv\Scripts\python.exe -m modal run .\modal_app.py::pack_corpus
```

Stage đọc 8.532 JSON và lưu:

```text
legalqa-data:/cache/corpus_metadata.pkl
```

#### Bước 3: build BM25 bằng CPU

```powershell
.\.venv\Scripts\python.exe -m modal run .\modal_app.py::build_bm25
```

Output:

```text
legalqa-data:/cache/bm25_index.pkl
```

#### Bước 4: build parent embeddings bằng H100

```powershell
.\.venv\Scripts\python.exe -m modal run .\modal_app.py::build_dense_or_child `
  --kind dense `
  --batch-size 64
```

Output:

```text
legalqa-data:/cache/parent_embeddings.pkl
```

Nếu hết VRAM, giảm `--batch-size` xuống `32` hoặc `16`.

#### Bước 5: build child embeddings bằng H100

```powershell
.\.venv\Scripts\python.exe -m modal run .\modal_app.py::build_dense_or_child `
  --kind child `
  --batch-size 64
```

Output:

```text
legalqa-data:/cache/child_embeddings.pkl
```

Parent và child embedding được lưu dạng NumPy `float32`; cache tạo bằng T4 vẫn có thể dùng trên H100.

#### Bước 6: kiểm tra ba base cache

```powershell
.\.venv\Scripts\python.exe -m modal run .\modal_app.py::inspect_inputs
```

Cần thấy `packed_corpus: true` và cả ba mục `base` đều là `true`.

#### Bước 7: tạo bốn feature cache

```powershell
.\.venv\Scripts\python.exe -m modal run .\modal_app.py::prepare_features `
  --run-id prepare-v21 `
  --ce-batch-size 32
```

Stage H100 này tạo hoặc tiếp tục:

```text
legalqa-data:/cache/top_documents.pkl
legalqa-data:/cache/candidate_audit.pkl
legalqa-data:/cache/pair_features.pkl
legalqa-data:/cache/singleton_features.pkl
legalqa-data:/cache/feature_cache_complete.json
```

Nếu bị dừng, chạy lại cùng lệnh. Cache lưu theo qid và chỉ xử lý phần chưa hoàn thành.

#### Bước 8: train selector V2.1 bằng CPU

```powershell
.\.venv\Scripts\python.exe -m modal run .\modal_app.py::train_selector_v21 `
  --run-id selector-v21-train
```

Đợi stage hoàn thành rồi kiểm tra:

```powershell
.\.venv\Scripts\python.exe -m modal run .\modal_app.py::inspect_inputs
```

Kết quả phải có `selector_ready: true`.

#### Bước 9: smoke test private

```powershell
.\.venv\Scripts\python.exe -m modal run .\modal_app.py `
  --dataset private `
  --private-limit 10 `
  --ce-batch-size 32 `
  --checkpoint-every 5
```

#### Bước 10: chạy full private và lấy file nộp

```powershell
.\.venv\Scripts\python.exe -m modal run .\modal_app.py `
  --dataset private `
  --private-limit 0 `
  --ce-batch-size 32 `
  --checkpoint-every 25
```

File nộp:

```text
outputs/modal/private-<id>/submission_private.zip
```

### Resume và tải kết quả thủ công

Nếu private inference bị ngắt, giữ nguyên `private-<id>` đã được in ở lần chạy trước:

```powershell
.\.venv\Scripts\python.exe -m modal run .\modal_app.py `
  --dataset private `
  --private-limit 0 `
  --resume-run-id private-<id> `
  --ce-batch-size 32 `
  --checkpoint-every 25
```

Checkpoint có chữ ký bundle. Nếu bundle model thay đổi, checkpoint cũ không được tái sử dụng.

Nếu Modal đã chạy xong nhưng terminal không tải output về:

```powershell
.\.venv\Scripts\python.exe -m modal volume get --force `
  legalqa-results "/private-<id>" ".\outputs\modal\private-<id>"
```

Có thể backup cache training và model bundle:

```powershell
.\.venv\Scripts\python.exe -m modal volume get --force `
  legalqa-data /cache .\data\cache

.\.venv\Scripts\python.exe -m modal volume get --force `
  legalqa-models /legalqa/legalqa_selector_v21 .\data\models\legalqa_selector_v21
```

Không xóa ba Modal Volume nếu muốn tái sử dụng cache, OOF, model và checkpoint cho lần sau.
## Phân bổ tài nguyên

| Giai đoạn | Tài nguyên | Lý do |
|---|---|---|
| Pack corpus, BM25 | CPU | Không dùng neural model |
| Parent/child embeddings | H100 | Batch embedding |
| Feature cache | H100 + CPU | Retrieval và CrossEncoder |
| Final targets | CPU | Answer builder và METEOR |
| OOF/full selector training | 32 CPU | XGBoost, LightGBM, CatBoost dạng bảng |
| Public/private inference | H100 | Query embedding và reranker |

Cache embedding tạo bằng T4 có thể đọc trên H100 vì dữ liệu lưu dưới dạng NumPy `float32`; code không khóa theo tên GPU.

## Kiểm tra đầu ra

Trước khi nộp, xác nhận file ZIP chỉ có `submission.json`, đủ qid, không có answer rỗng và inference log không có lỗi:

```powershell
$zip = Get-ChildItem .\outputs\modal -Recurse -Filter submission_private.zip |
  Sort-Object LastWriteTime -Descending | Select-Object -First 1

.\.venv\Scripts\python.exe -c "import json,zipfile,sys; z=zipfile.ZipFile(sys.argv[1]); assert z.namelist()==['submission.json']; d=json.loads(z.read('submission.json')); assert d and all(str(v.get('answer','')).strip() for v in d.values()); print('PASS',len(d),'qids')" $zip.FullName
```

## Test

```powershell
.\.venv\Scripts\python.exe -m unittest discover -s tests -v
```

Test bao phủ ACTION40/15 actions, transforms 36/109 features, safety mask, pairwise symmetry, Borda, gate `0.55`, native bundle, post-processing và định dạng submission ZIP.
