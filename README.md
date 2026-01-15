# RAG – Hệ thống hỏi đáp Luật Giao Thông (PDF → Chunk → VectorDB → Retrieval → LLM → UI)
#### Soạn bởi: Bùi Quang Thái
Dự án này xây dựng một hệ thống **Retrieval-Augmented Generation (RAG)** để hỏi–đáp dựa trên **tài liệu PDF tiếng Việt** (ví dụ: Luật giao thông).  
Hệ thống hỗ trợ:

- ✅ Hỏi–đáp theo nội dung PDF (RAG)
- ✅ Hiển thị **Top-k context + similarity score** để kiểm chứng
- ✅ Danh sách tài liệu đã nạp + mở/tải PDF trong UI
- ✅ Upload PDF mới và **ingest incremental** (chỉ nạp file mới, chống trùng bằng hash)
- ✅ VectorDB **persist** (lưu bền) bằng **ChromaDB**

---

## 1. Kiến trúc tổng quan

### Pipeline RAG
```bash
PDFs (Document Store)
│
├─ Load (PyPDFLoader)
│
├─ Clean Vietnamese text (Unicode normalize + remove noise)
│
├─ Chunking (RecursiveCharacterTextSplitter)
│
├─ Embedding (HuggingFace / Sentence-Transformers)
│
├─ VectorDB (ChromaDB persist)
│
User Query
│
├─ Retrieval (Similarity Search Top-k)
│
├─ (Optional) show Top-k + similarity score
│
├─ Prompt Template (Context + Question)
│
└─ LLM (Qwen) → Answer
```
### Luồng “thêm tài liệu” (incremental ingest)
Upload PDF → Copy vào kho → Hash check (chống trùng)
→ Load → Clean → Chunk → Add to Chroma
→ Refresh Retriever/Chain → Sẵn sàng QA ngay

---

## 2. Cấu trúc thư mục dự án

```bash

rag_langchain/
├─ data_source/
│ ├─ generative_ai/ # kho PDF chính (bạn copy PDF vào đây)
│ └─ custom/ # tuỳ chọn (bộ PDF khác)
├─ chroma_data/ # nơi ChromaDB persist Vector Database
├─ ingested_hashes.json # log hash để chống ingest trùng
└─ src/
├─ base/init.py
└─ rag/init.py
```
### Ý nghĩa từng thành phần
- `data_source/generative_ai/`: **Document Store** chứa PDF gốc (nguồn tri thức thô).
- `chroma_data/`: **Vector Store** (vector embeddings + documents + metadata + index).
- `ingested_hashes.json`: lưu hash từng file PDF đã ingest để **skip trùng**.
- `src/`: chuẩn hoá cấu trúc module (dễ tách notebook thành project Python).

---

## 4. Cài đặt môi trường

### 4.1 Yêu cầu
- Python: 3.10–3.12 (khuyến nghị 3.10/3.11 để tương thích rộng)
- GPU: không bắt buộc (CPU vẫn chạy được, nhưng chậm hơn)
- OS: Windows / Linux / macOS

### 4.2 Cài dependencies (tham khảo)
> Bạn có thể gom thành `requirements.txt`. Ví dụ:

```bash
pip install -U pip

pip install \
  "torch" \
  "transformers>=4.40.0" \
  "accelerate>=0.30.0" \
  "sentence-transformers>=2.7.0" \
  "langchain>=0.2.0" \
  "langchain-core>=0.2.0" \
  "langchain-community>=0.2.0" \
  "langchain-text-splitters>=0.2.0" \
  "chromadb>=0.5.0" \
  "langchain-chroma>=0.2.0" \
  "langchain-huggingface" \
  "pypdf>=4.2.0" \
  "gradio>=4.0.0" \
  "pymupdf"
```

### 5. Tiền xử lý văn bản & Chunking
#### 5.1. Làm sạch tiếng Việt (clean text)

Mục tiêu:

Chuẩn hoá Unicode (NFC) để giảm lỗi dấu tiếng Việt

Loại ký tự control rác

Gom khoảng trắng thừa, dòng thừa

#### 5.2. Chunking

Dùng RecursiveCharacterTextSplitter:

chunk_size=400

chunk_overlap=120

separators: ["\n\n", "\n", " ", ""]

Lý do:

Chunk nhỏ giúp embedding bắt nghĩa tốt

Overlap giữ ngữ cảnh ở ranh giới Điều/Khoản

### 6. Vector Database (ChromaDB) lưu như thế nào?

ChromaDB persist tại:

rag_langchain/chroma_data/


VectorDB lưu:

Embedding vectors (vector float cho từng chunk)

Document text (nội dung chunk)

Metadata (source_file, page, …)

Index phục vụ similarity search nhanh

⚠️ Quan trọng: Nếu đổi embedding model, bạn cần:

đổi collection_name hoặc

xoá chroma_data/ và rebuild
vì vector space của model khác nhau.

### 7. Retrieval + Similarity score (kiểm chứng)

Hệ thống hỗ trợ lấy context theo 2 cách:

Retriever chuẩn của LangChain (Top-k context)

Chroma similarity_search_with_score để hiển thị score

Ví dụ format context:
```bash
[file.pdf | page 12 | score=0.1234]
<nội dung chunk ...>
```

Lưu ý: với Chroma, score thường là distance (thấp hơn = giống hơn).

### 8. Giao diện demo (Gradio)

UI gồm:

Cột trái:

Input câu hỏi

Output câu trả lời

Output Top-k context + score

Cột phải:

Danh sách PDF đã nạp

Mở/tải PDF

Upload PDF mới + ingest incremental

Tab “Phương pháp” (giải thích pipeline)


### 9. Kiểm tra tính chính xác (Evaluation)

Đánh giá theo 3 tầng:

#### 9.1. Đánh giá Retrieval (quan trọng nhất)

Gợi ý metrics:

Precision@k

Recall@k

MRR

nDCG@k (nếu có mức độ liên quan)

Cách tạo ground truth nhanh:

Tạo 30–100 câu hỏi

Với mỗi câu, gán nhãn “chunk/trang đúng” (ít nhất 1 chunk chứa đáp án)

Tính metrics theo retrieval results

#### 9.2. Đánh giá câu trả lời (Groundedness)

Chấm theo rubric 0–2:

2: hoàn toàn bám context

1: suy diễn nhẹ

0: hallucination (bịa)

#### 9.3. End-to-end (Human evaluation)

Task success

Điểm 1–5 theo: đúng / rõ / hữu ích / có dẫn chứng

### 10. Ưu điểm & Nhược điểm
Ưu điểm

Explainable: hiển thị Top-k context + score + file/page

Dễ mở rộng: upload PDF, ingest incremental

Chi phí thấp: chạy local/1 máy vẫn demo được

Tách bạch rõ: PDF store ≠ Vector store

Nhược điểm

PDF scan ảnh → loader có thể trích text rỗng (cần OCR)

Chunking theo ký tự có thể cắt sai Điều/Khoản (nếu PDF phức tạp)

Không có re-ranking → đôi khi Top-k chưa tối ưu

LLM vẫn có rủi ro “bịa” nếu prompt/guard chưa chặt

### 11. Hướng nâng cấp (Advanced)

Hybrid Search: BM25 + Vector (giảm fail câu hỏi có keyword pháp lý)

Re-ranking: Cross-Encoder / bge-reranker để lọc top-k tốt hơn

Chunk theo cấu trúc luật: tách Điều/Khoản/Điểm thay vì ký tự

OCR pipeline: ocrmypdf / easyocr / tesseract cho PDF scan

Citations chuẩn: output kèm trích dẫn [file|page|chunk_id]

Production architecture: tách Ingest service & Query service (FastAPI), logging, eval harness

### 12. Lưu ý khi đổi Embedding model (trust_remote_code)

Một số model (ví dụ dangvantuan/vietnamese-document-embedding) có custom code trên HuggingFace → cần bật:
```bash
HuggingFaceEmbeddings(
  model_name="dangvantuan/vietnamese-document-embedding",
  model_kwargs={"trust_remote_code": True}
)
```

Và nhớ rebuild DB (hoặc đổi collection) như mục (6).

### 13. Demo checklist (gợi ý thuyết trình)

Chọn PDF → mở/tải hoặc preview nhiều trang

Search keyword → nhảy đúng trang & highlight

Đặt câu hỏi → xem Answer + Top-k context + score

Giải thích pipeline: Load → Chunk → VectorDB → Retrieval → LLM → UI