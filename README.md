# RepPYQ — SPPU Question Paper Analyzer & Repeated PYQ Extractor

RepPYQ helps university students (built for **SPPU 2019 Pattern**) upload past exam PDFs, automatically extract exact verbatim questions, map them to **Unit 1–2 (Insem)** or **Unit 3–6 (Endsem)**, group rephrased repeated questions using semantic embeddings + FAISS, and export clean **Unit-Wise Revision PDFs**.

Built by **[Pushkar Shinde](https://pushkarez.vercel.app/)**

## Tech Stack
- **Frontend:** React, TypeScript, Vite, Tailwind CSS
- **Backend:** FastAPI, PyMuPDF (`fitz`), Tesseract OCR fallback, `BAAI/bge-small-en-v1.5`, `faiss-cpu`, `scikit-learn`, `ReportLab`
