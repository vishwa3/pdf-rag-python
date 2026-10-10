import difflib
import json
import os
from contextlib import asynccontextmanager

from dotenv import load_dotenv
from fastapi import FastAPI, File, HTTPException, Request, UploadFile
from fastapi.responses import StreamingResponse
from fastapi.staticfiles import StaticFiles
from google import genai
from langsmith import traceable
from psycopg.rows import dict_row
from psycopg_pool import AsyncConnectionPool
from pydantic import BaseModel

from ingest import ingest_pdf

load_dotenv()


@asynccontextmanager
async def lifespan(app: FastAPI):
    # Startup: create and open pool
    app.state.pool = AsyncConnectionPool(
        kwargs=DB_CONFIG, min_size=1, max_size=4, open=False
    )
    await app.state.pool.open()
    app.state.ai = genai.Client(api_key=os.getenv("GOOGLE_API_KEY"))

    async with app.state.pool.connection() as conn:
        await conn.execute("CREATE EXTENSION IF NOT EXISTS vector;")
        await conn.execute(
            """
            CREATE TABLE IF NOT EXISTS ingested_files (
                id SERIAL PRIMARY KEY,
                file_path TEXT UNIQUE NOT NULL,
                file_hash VARCHAR(64) NOT NULL,
                updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            )
            """
        )

        await conn.execute(
            "CREATE INDEX IF NOT EXISTS idx_ingested_files_hash ON ingested_files(file_hash);"
        )
        await conn.execute(
            """
            CREATE TABLE IF NOT EXISTS document_vectors (
                id SERIAL PRIMARY KEY,
                content TEXT NOT NULL,
                metadata JSONB,
                embedding vector(3072)
            )
            """
        )
    yield
    await app.state.pool.close()


app = FastAPI(title="PDF RAG PYTHON", lifespan=lifespan)


class ChatRequest(BaseModel):
    question: str


class Source(BaseModel):
    source: str
    chunk_index: int
    distance: float


class ChatResponse(BaseModel):
    answer: str
    sources: list[Source]


DB_CONFIG = {
    "host": "localhost",
    "port": os.getenv("DB_PORT"),
    "user": os.getenv("DB_USER"),
    "password": os.getenv("DB_PASSWORD"),
    "dbname": os.getenv("DB_NAME"),
    "row_factory": dict_row,
}


@app.post("/api/chat")
async def chat(http_request: Request, request: ChatRequest) -> ChatResponse:
    pool: AsyncConnectionPool = http_request.app.state.pool

    # 1. Embed the user's question -> 3072-dim vector

    ai: genai.Client = http_request.app.state.ai
    embedding_result = await ai.aio.models.embed_content(
        model="gemini-embedding-2", contents=request.question
    )
    embeddings = embedding_result.embeddings
    if not embeddings or embeddings[0].values is None:
        raise HTTPException(
            status_code=502, detail="Gemini returned no embeddings for the question."
        )
    user_question_vector = embeddings[0].values

    # 2. Query pgvector for the most similar chunks
    async with pool.connection() as conn:
        cur = await conn.execute(
            """
            SELECT content, metadata, embedding <=> %s::vector AS distance
            FROM document_vectors
            ORDER BY distance
            LIMIT 12
            """,
            (str(user_question_vector),),
        )
        rows: list[dict] = await cur.fetchall()  # type: ignore[assignment]

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

    # 3. Build context from the retrieved chunks
    context = "\n---\n".join(row["content"] for row in selected)

    # 4. Generate a grounded answer
    response = await ai.aio.models.generate_content(
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


async def _stream_answer(http_request: Request, question: str):
    """ASYNC generator: yields SSE events — sources first, then tokens."""
    pool: AsyncConnectionPool = http_request.app.state.pool

    # 1. Embed the user's question -> 3072-dim vector
    ai = http_request.app.state.ai
    embedding_result = await ai.aio.models.embed_content(
        model="gemini-embedding-2", contents=question
    )
    embeddings = embedding_result.embeddings
    if not embeddings or embeddings[0].values is None:
        yield f"data: {json.dumps({'error': 'Failed to embed question'})}\n\n"
        return
    user_question_vector = embeddings[0].values

    # 2. Query pgvector for the most similar chunks (LIMIT 12 like /api/chat)
    async with pool.connection() as conn:
        cur = await conn.execute(
            """
            SELECT content, metadata, embedding <=> %s::vector AS distance
            FROM document_vectors
            ORDER BY distance
            LIMIT 12
            """,
            (str(user_question_vector),),
        )
        rows: list[dict] = await cur.fetchall()  # type: ignore[assignment]

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
        stream = await ai.aio.models.generate_content_stream(
            model="gemini-2.5-flash", contents=prompt, config={"temperature": 0.1}
        )

        async for chunk in stream:
            if chunk.text:
                yield f"data: {json.dumps({'token': chunk.text})}\n\n"

        # 7. Signal the end
        yield "data: [DONE]\n\n"

    except Exception as e:
        yield f"data: {json.dumps({'error': str(e)})}\n\n"


@app.post("/api/chat/stream")
async def chat_stream(http_request: Request, request: ChatRequest) -> StreamingResponse:
    return StreamingResponse(
        _stream_answer(http_request, request.question), media_type="text/event-stream"
    )


# Serve the frontend (must be LAST — it's a catch-all for everything unmatched)
app.mount("/", StaticFiles(directory="static", html=True), name="static")
