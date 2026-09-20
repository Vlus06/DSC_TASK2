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

Đăng nhập và tạo ba Volume một lần:

```powershell
.\.venv\Scripts\python.exe -m pip install modal
.\.venv\Scripts\python.exe -m modal setup
.\.venv\Scripts\python.exe -m modal volume create legalqa-data
.\.venv\Scripts\python.exe -m modal volume create legalqa-models
.\.venv\Scripts\python.exe -m modal volume create legalqa-results
```

Nạp dataset, stopwords và cache có sẵn:

```powershell
.\.venv\Scripts\python.exe -m modal volume put --force legalqa-data .\data\TASK2 /TASK2
.\.venv\Scripts\python.exe -m modal volume put --force legalqa-data .\data\stopwords.txt /stopwords.txt
.\.venv\Scripts\python.exe -m modal volume put --force legalqa-data .\data\cache /cache
```

Nếu chưa có `corpus_metadata.pkl`, tạo bằng CPU:

```powershell
.\.venv\Scripts\python.exe -m modal run .\modal_app.py::pack_corpus
```

Kiểm tra trạng thái:

```powershell
.\.venv\Scripts\python.exe -m modal run .\modal_app.py::inspect_inputs
```

Nếu bốn feature cache đã có nhưng chưa có bundle V2.1, train riêng trên CPU 32 cores:

```powershell
.\.venv\Scripts\python.exe -m modal run .\modal_app.py::train_selector_v21 `
  --run-id selector-v21-train
```

Final-target và OOF cache được giữ trên `legalqa-data:/cache`. Bundle được giữ tại:

```text
legalqa-models:/legalqa/legalqa_selector_v21/
```

Tải bundle về để backup:

```powershell
.\.venv\Scripts\python.exe -m modal volume get --force `
  legalqa-models /legalqa/legalqa_selector_v21 .\data\models\legalqa_selector_v21
```

Chạy private inference bằng H100 từ bundle đã train:

```powershell
.\.venv\Scripts\python.exe -m modal run .\modal_app.py `
  --dataset private `
  --private-limit 0 `
  --ce-batch-size 32 `
  --checkpoint-every 25
```

Smoke test 10 câu dùng `--private-limit 10`. Khi chạy full, file nộp được tải về:

```text
outputs/modal/private-<run_id>/submission_private.zip
```

ZIP chỉ chứa một file có tên `submission.json`. JSON rời và inference log cũng được lưu cùng thư mục. Nếu terminal bị ngắt, dùng lại run id:

```powershell
.\.venv\Scripts\python.exe -m modal run .\modal_app.py `
  --dataset private `
  --private-limit 0 `
  --resume-run-id private-<run_id>
```

Hoặc tải output trực tiếp từ Volume:

```powershell
.\.venv\Scripts\python.exe -m modal volume get --force `
  legalqa-results "/private-<run_id>" ".\outputs\modal\private-<run_id>"
```

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
