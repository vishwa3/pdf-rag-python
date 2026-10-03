import difflib
import json
import os
from contextlib import asynccontextmanager

from dotenv import load_dotenv
from fastapi import FastAPI, File, HTTPException, Request, UploadFile
from fastapi.responses import StreamingResponse
from google import genai
from langsmith import traceable
from psycopg import Connection
from psycopg.rows import DictRow, dict_row
from psycopg_pool import ConnectionPool
from pydantic import BaseModel
from fastapi.staticfiles import StaticFiles

from ingest import ensure_schema, ingest_pdf

load_dotenv()


@asynccontextmanager
async def lifespan(app: FastAPI):
    # Startup: create and open pool
    app.state.pool = ConnectionPool[Connection[DictRow]](
        kwargs=DB_CONFIG, min_size=1, max_size=4, open=True
    )
    with app.state.pool.connection() as conn:
        ensure_schema(conn)
    yield
    # Shutdown: close pool
    app.state.pool.close()


app = FastAPI(title="PDF RAG PYTHON", lifespan=lifespan)


class ChatRequest(BaseModel):
    question: str


class Source(BaseModel):
    source: str
    chunk_index: int
    distance: float


class ChatResponse(BaseModel):
    answer: str
    sources: list[Source] = []


DB_CONFIG = {
    "host": "localhost",
    "port": os.getenv("DB_PORT"),
    "user": os.getenv("DB_USER"),
    "password": os.getenv("DB_PASSWORD"),
    "dbname": os.getenv("DB_NAME"),
    "row_factory": dict_row,
}


@app.post("/api/chat")
def chat(http_request: Request, request: ChatRequest) -> ChatResponse:
    pool: ConnectionPool[Connection[DictRow]] = http_request.app.state.pool

    # 1. Embed the user's question -> 3072-dim vector

    ai = genai.Client(api_key=os.getenv("GOOGLE_API_KEY"))
    embedding_result = ai.models.embed_content(
        model="gemini-embedding-2", contents=request.question
    )
    embeddings = embedding_result.embeddings
    if not embeddings or embeddings[0].values is None:
        raise HTTPException(
            status_code=502, detail="Gemini returned no embeddings for the question."
        )
    user_question_vector = embeddings[0].values

    # 2. Query pgvector for the most similar chunks
    with pool.connection() as conn:
        rows = conn.execute(
            """
            SELECT content, metadata, embedding <=> %s::vector AS distance
            FROM document_vectors
            ORDER BY distance
            LIMIT 12
            """,
            (str(user_question_vector),),
        ).fetchall()
        print(rows)

    # 2b. Keep the best chunk, skipping near-duplicates: different versions of
    # the same letter crowd the top-3 with near-identical chunks otherwise.
    selected: list[dict] = []
    for row in rows:
        if any(
            difflib.SequenceMatcher(
                None, row["content"][:400], kept["content"][:400]
            ).ratio()
            >= 0.85
            for kept in selected
        ):
            continue
        selected.append(row)
        if len(selected) == 3:
            break
    print(
        f" Selected {len(selected)} chunks after filtering for near-duplicates",
        selected,
    )

    # 3. Build context from the retrieved chunks
    context = "\n---\n".join(row["content"] for row in selected)

    # 4. Generate a grounded answer
    response = ai.models.generate_content(
        model="gemini-2.5-flash",
        contents=(
            "You are a precise assistant. Answer the user's question using ONLY the "
            "facts found within the context snippets below.\n"
            "If the answer cannot be confidently derived from the provided snippets, "
            'respond exactly with: "I cannot find that information inside the uploaded document."\n'
            "Do not make up facts.\n\n"
            "Context Snippets:\n"
            "=========================================\n"
            f"{context}\n"
            "=========================================\n\n"
            f"User Question: {request.question}\n"
            "Answer:"
        ),
        config={"temperature": 0.1},
    )

    sources = [
        Source(
            source=row["metadata"]["source"],
            chunk_index=row["metadata"]["chunkIndex"],
            distance=row["distance"],
        )
        for row in selected
    ]

    return ChatResponse(answer=response.text or "", sources=sources)


@app.post("/api/ingest")
def ingest_document(file: UploadFile = File(...)) -> dict:
    # 1. Validate it's a PDF
    filename = file.filename or "upload.pdf"
    if not filename.lower().endswith(".pdf"):
        raise HTTPException(status_code=400, detail="Only PDF files are supported")

    # 2. Read the entire uploaded file into memory
    file_bytes = file.file.read()
    chunk_count = ingest_pdf(file_bytes, f"docs/{filename}")
    return {"status": "ok", "file": filename, "chunks": chunk_count}


def _stream_answer(http_request: Request, question: str):
    pool: ConnectionPool[Connection[DictRow]] = http_request.app.state.pool

    # 1. Embed the user's question -> 3072-dim vector
    ai = genai.Client(api_key=os.getenv("GOOGLE_API_KEY"))
    embedding_result = ai.models.embed_content(
        model="gemini-embedding-2", contents=question
    )
    embeddings = embedding_result.embeddings
    if not embeddings or embeddings[0].values is None:
        yield f"data: {json.dumps({'error': 'Failed to embed question'})}\n\n"
        return
    user_question_vector = embeddings[0].values

    # 2. Query pgvector for the most similar chunks (LIMIT 12 like /api/chat)
    with pool.connection() as conn:
        rows = conn.execute(
            """
            SELECT content, metadata, embedding <=> %s::vector AS distance
            FROM document_vectors
            ORDER BY distance
            LIMIT 12
            """,
            (str(user_question_vector),),
        ).fetchall()

    # 3. Keep 3 best chunks, skipping near-duplicates (same logic as /api/chat)
    selected: list[dict] = []
    for row in rows:
        if any(
            difflib.SequenceMatcher(
                None, row["content"][:400], kept["content"][:400]
            ).ratio()
            >= 0.85
            for kept in selected
        ):
            continue
        selected.append(row)
        if len(selected) == 3:
            break

    # 4. Send SOURCES first — the UI can show "found 3 chunks" instantly

    sources = [
        Source(
            source=row["metadata"]["source"],
            chunk_index=row["metadata"]["chunkIndex"],
            distance=row["distance"],
        )
        for row in selected
    ]

    yield f"data: {json.dumps({'sources': [s.model_dump() for s in sources]})}\n\n"

    # 5. Build the grounded prompt
    context = "\n---\n".join(row["content"] for row in selected)

    prompt = (
        "You are a precise assistant. Answer the user's question using ONLY the "
        "facts found within the context snippets below.\n"
        "If the answer cannot be confidently derived from the provided snippets, "
        'respond exactly with: "I cannot find that information inside the uploaded document."\n'
        "Do not make up facts.\n\n"
        "Context Snippets:\n"
        "=========================================\n"
        f"{context}\n"
        "=========================================\n\n"
        f"User Question: {question}\n"
        "Answer:"
    )

    # 6. Stream tokens as they're generated
    try:
        stream = ai.models.generate_content_stream(
            model="gemini-2.5-flash", contents=prompt, config={"temperature": 0.1}
        )

        for chunk in stream:
            if chunk.text:
                yield f"data: {json.dumps({'token': chunk.text})}\n\n"

        # 7. Signal the end
        yield "data: [DONE]\n\n"

    except Exception as e:
        yield f"data: {json.dumps({'error': str(e)})}\n\n"


@app.post("/api/chat/stream")
def chat_stream(http_request: Request, request: ChatRequest) -> StreamingResponse:
    return StreamingResponse(
        _stream_answer(http_request, request.question), media_type="text/event-stream"
    )


# Serve the frontend (must be LAST — it's a catch-all for everything unmatched)
app.mount("/", StaticFiles(directory="static", html=True), name="static")
