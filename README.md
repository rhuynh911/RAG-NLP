# RAG – Hệ thống hỏi đáp Luật Giao Thông (PDF → Chunk → VectorDB → Retrieval → LLM → UI)

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

### Luồng “thêm tài liệu” (incremental ingest)
Upload PDF → Copy vào kho → Hash check (chống trùng)
→ Load → Clean → Chunk → Add to Chroma
→ Refresh Retriever/Chain → Sẵn sàng QA ngay

---

## 2. Cấu trúc thư mục dự án
rag_langchain/
├─ data_source/
│ ├─ generative_ai/ # kho PDF chính (bạn copy PDF vào đây)
│ └─ custom/ # tuỳ chọn (bộ PDF khác)
├─ chroma_data/ # nơi ChromaDB persist Vector Database
├─ ingested_hashes.json # log hash để chống ingest trùng
└─ src/
├─ base/init.py
└─ rag/init.py

### Ý nghĩa từng thành phần
- `data_source/generative_ai/`: **Document Store** chứa PDF gốc (nguồn tri thức thô).
- `chroma_data/`: **Vector Store** (vector embeddings + documents + metadata + index).
- `ingested_hashes.json`: lưu hash từng file PDF đã ingest để **skip trùng**.
- `src/`: chuẩn hoá cấu trúc module (dễ tách notebook thành project Python).

---

## 4) Cài đặt môi trường

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
'''

sfdgdf