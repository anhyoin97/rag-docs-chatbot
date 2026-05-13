from pathlib import Path
import os
import uuid

from dotenv import load_dotenv
from flask import Flask, render_template, redirect, request, url_for
import chromadb # 로컬에 DB 폴더를 두고 벡터 저장
from openai import OpenAI # OpenAI API 호출

load_dotenv()

CHROMA_DIR = Path(__file__).resolve().parent / "chroma_data"
EMBEDDING_MODEL = "text-embedding-3-small"
EMBEDDING_BATCH_SIZE = 64
CHROMA_DIR.mkdir(exist_ok=True)

app = Flask(__name__)  # Flask 애플리케이션(서버) 객체 생성

chroma_client = chromadb.PersistentClient(path=str(CHROMA_DIR))
chroma_collection = chroma_client.get_or_create_collection(
    name="rag_mvp",
    metadata={"hnsw:space": "cosine"},
)

# __file__ : 현재 파일의 경로 
# Path(__file__) : 그 문자열 경로를 pathlib.Path 객체로 변환
# .resolve() : 현재 파일의 절대 경로를 반환 
# .parent : 현재 파일의 부모 디렉토리 경로를 반환
BASE_DIR = Path(__file__).resolve().parent   # 프로젝트의 루트 폴더
UPLOAD_DIR = BASE_DIR / "uploads"
UPLOAD_DIR.mkdir(exist_ok=True)


def chunk_text(text: str, max_chars: int = 800, overlap: int = 100) -> list[str]:
    """문자 수 기준으로 청크를 나눈다. MVP용 단순 슬라이딩 윈도우."""
    text = text.strip()
    if not text:
        return []
    chunks: list[str] = []
    start = 0
    n = len(text)
    while start < n:
        end = min(start + max_chars, n)
        piece = text[start:end].strip()
        if piece:
            chunks.append(piece)
        if end >= n:
            break
        start = end - overlap
        if start < 0:
            start = 0
    return chunks

@app.get("/") # / 경로로 GET 요청이 오면 아래 함수 실행 (라우팅)
def home():
    files = sorted([p.name for p in UPLOAD_DIR.glob("*.txt")])
    has_api_key = bool(os.getenv("OPENAI_API_KEY"))
    return render_template("index.html", files=files, has_api_key=has_api_key)

@app.post("/upload")
def upload_file():
    file = request.files.get("file")
    if file is None or file.filename == "":
        return redirect(url_for("home"))
    safe_name = Path(file.filename).name
    if not safe_name.lower().endswith(".txt"):
        return redirect(url_for("home"))
    save_path = UPLOAD_DIR / safe_name
    file.save(save_path)
    return redirect(url_for("home"))


@app.get("/index")
def index_docs():
    """uploads의 .txt를 청크 → OpenAI 임베딩 → Chroma에 저장(전체 재인덱싱)."""
    stats: list[dict] = []
    records: list[dict] = []

    for path in sorted(UPLOAD_DIR.glob("*.txt")):
        text = path.read_text(encoding="utf-8")
        chunks = chunk_text(text)
        stats.append({"filename": path.name, "chunks": len(chunks)})
        for i, chunk in enumerate(chunks):
            records.append(
                {
                    "source": path.name,
                    "chunk_index": i,
                    "text": chunk,
                }
            )

    total_chunks = len(records)
    file_count = len(stats)
    vectors_in_db = chroma_collection.count()
    indexed = False
    error_message: str | None = None

    def render():
        return render_template(
            "indexing.html",
            stats=stats,
            total_chunks=total_chunks,
            file_count=file_count,
            vectors_in_db=vectors_in_db,
            indexed=indexed,
            error_message=error_message,
        )

    if not os.getenv("OPENAI_API_KEY"):
        error_message = "OPENAI_API_KEY가 없습니다. .env를 확인하세요."
        return render()

    try:
        existing = chroma_collection.get()
        if existing and existing.get("ids"):
            chroma_collection.delete(ids=existing["ids"])

        if total_chunks == 0:
            vectors_in_db = chroma_collection.count()
            indexed = True
            return render()

        client = OpenAI()
        all_embeddings: list[list[float]] = []

        texts = [r["text"] for r in records]
        for start in range(0, len(texts), EMBEDDING_BATCH_SIZE):
            batch = texts[start : start + EMBEDDING_BATCH_SIZE]
            resp = client.embeddings.create(model=EMBEDDING_MODEL, input=batch)
            ordered = sorted(resp.data, key=lambda d: d.index)
            all_embeddings.extend(d.embedding for d in ordered)

        ids = [str(uuid.uuid4()) for _ in records]
        metadatas = [
            {"source": r["source"], "chunk_index": r["chunk_index"]} for r in records
        ]

        chroma_collection.add(
            ids=ids,
            embeddings=all_embeddings,
            documents=texts,
            metadatas=metadatas,
        )
        indexed = True
        vectors_in_db = chroma_collection.count()
    except Exception as exc:  # noqa: BLE001 — MVP: 사용자에게 원인 표시
        error_message = str(exc)
        vectors_in_db = chroma_collection.count()

    return render()


if __name__ == "__main__":
    app.run(debug=True)