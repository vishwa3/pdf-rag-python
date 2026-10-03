import hashlib
import json
import os
from pathlib import Path
from typing import Any, cast

from dotenv import load_dotenv
from google import genai
from langsmith import traceable
from psycopg import Connection, connect
from psycopg.rows import DictRow, dict_row

from chunking import markdown_aware_chunk

load_dotenv()

api_key = os.getenv("GOOGLE_API_KEY")

if not api_key:
    raise ValueError("❌ GOOGLE_API_KEY is not defined in your .env file!")

client = genai.Client(api_key=api_key)

FILE_PATH = os.getenv("PDF_FILE_PATH", "docs/Generative AI Primer - Bocconi.pdf")

DB_CONFIG = {
    "host": "localhost",
    "port": os.getenv("DB_PORT"),
    "connect_timeout": 5,  # fail fast if Postgres is down instead of hanging silently
    "user": os.getenv("DB_USER"),
    "password": os.getenv("DB_PASSWORD"),
    "dbname": os.getenv("DB_NAME"),
    "row_factory": dict_row,
}


def ensure_schema(conn: Connection[Any]) -> None:
    """Create the pgvector extension and both tables if they don't exist."""
    conn.execute("CREATE EXTENSION IF NOT EXISTS vector;")

    conn.execute(
        """
        CREATE TABLE IF NOT EXISTS ingested_files (
            id SERIAL PRIMARY KEY,
            file_path TEXT UNIQUE NOT NULL,
            file_hash VARCHAR(64) NOT NULL,
            updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        )
        """
    )

    conn.execute(
        "CREATE INDEX IF NOT EXISTS idx_ingested_files_hash ON ingested_files(file_hash);"
    )

    conn.execute(
        """
        CREATE TABLE IF NOT EXISTS document_vectors (
                id SERIAL PRIMARY KEY,
                content TEXT NOT NULL,
                metadata JSONB,
                embedding vector(3072)
        )
        """
    )


def ingest_pdf(file_bytes: bytes, file_path: str) -> int:
    """
    Ingest ONE PDF: extract -> chunk -> embed -> store.
    Returns the number of chunks stored (0 if skipped/unchanged).
    """
    with cast(Connection[DictRow], connect(**DB_CONFIG)) as conn:
        ensure_schema(conn)

        # 1. Hash to detect changes
        current_file_hash = hashlib.sha256(file_bytes).hexdigest()

        # 2. Does this path exist, and did it change?
        existing_file = conn.execute(
            "SELECT file_hash FROM ingested_files WHERE file_path = %s", (file_path,)
        ).fetchone()

        if existing_file:
            if existing_file["file_hash"] == current_file_hash:
                print(f"[SKIP] {file_path} has not changed")
                return 0
            else:
                print(f"[UPDATE] {file_path} was modified. Purging old vectors...")
                conn.execute(
                    "DELETE FROM document_vectors WHERE (metadata->>'source') = %s",
                    (file_path,),
                )

        # 3. Check if ANY other file already has this exact content
        identical_file = conn.execute(
            "SELECT file_path FROM ingested_files WHERE file_hash = %s AND file_path <> %s",
            (current_file_hash, file_path),
        ).fetchone()

        if identical_file:
            print(
                f"[COPY] Identical content found in {identical_file['file_path']}. Copying vectors ..."
            )
            conn.execute(
                """
                INSERT INTO document_vectors (content, metadata, embedding)
                SELECT content, jsonb_set(metadata, '{source}', to_jsonb(%s)), embedding
                FROM document_vectors WHERE (metadata->>'source') = %s
                """,
                (file_path, identical_file["file_path"]),
            )
            # Update tracking table and exit!
            conn.execute(
                """
                INSERT INTO ingested_files (file_path, file_hash, updated_at)
                VALUES (%s, %s, CURRENT_TIMESTAMP)
                ON CONFLICT (file_path)
                DO UPDATE SET file_hash = EXCLUDED.file_hash, updated_at = CURRENT_TIMESTAMP
                """,
                (file_path, current_file_hash),
            )
            return 0
        else:
            print(f"[NEW] Ingesting {file_path}")

        # 4. Extract -> chunk
        extracted_text = _extract_pdf_to_markdown(file_bytes)
        print(f"Extracted {len(extracted_text)} characters from PDF.")
        chunks = markdown_aware_chunk(extracted_text, 1000, 200)
        print(f"Split into {len(chunks)} table-aware chunks.")

        # 5. Embed each chunk and store in pgvector

        for i, chunk in enumerate(chunks):
            embedding_values = _embed_text(chunk)
            conn.execute(
                """
                INSERT INTO document_vectors (content, metadata, embedding)
                VALUES (%s, %s, %s::vector)
                """,
                (
                    chunk,
                    json.dumps({"source": file_path, "chunkIndex": i}),
                    json.dumps(embedding_values),
                ),
            )
        print(f"Successfully embedded and stored {len(chunks)} chunks!")

        # 6. Register/update the hash
        conn.execute(
            """
            INSERT INTO ingested_files (file_path, file_hash, updated_at)
            VALUES (%s, %s, CURRENT_TIMESTAMP)
            ON CONFLICT (file_path)
            DO UPDATE SET file_hash = EXCLUDED.file_hash, updated_at = CURRENT_TIMESTAMP
            """,
            (file_path, current_file_hash),
        )

        return len(chunks)


@traceable(name="Gemini PDF Extraction")
def _extract_pdf_to_markdown(file_bytes: bytes) -> str:
    """Send the PDF to Gemini Vision and get clean Markdown back."""
    response = client.models.generate_content(
        model="gemini-2.5-flash",
        contents=[
            genai.types.Part.from_bytes(data=file_bytes, mime_type="application/pdf"),
            """Extract ALL content from this PDF into clean Markdown, faithfully
            and completely. Preserve headings using proper Markdown syntax (# ## ###).
            Preserve lists, bullet points, and bold text.
            Convert ALL tables into clean Markdown table format (| col1 | col2 |).
            Do NOT summarize, skip any section, or add meta commentary.
            Output ONLY the raw extracted Markdown, nothing else.
            """,
        ],
    )

    return response.text or ""


@traceable(name="Gemini Chunk Embedding", run_type="embedding")
def _embed_text(text: str) -> list[float]:
    """Turn one chunk into a 3072-dimensional embedding vector."""
    embedding_result = client.models.embed_content(
        model="gemini-embedding-2", contents=text
    )
    embeddings = embedding_result.embeddings
    if not embeddings or embeddings[0].values is None:
        raise ValueError("Gemini returned no embeddings for the chunk.")
    return embeddings[0].values


def ingest() -> None:
    """Run the full ingestion pipeline."""

    file_bytes = Path(FILE_PATH).read_bytes()
    chunk_count = ingest_pdf(file_bytes, FILE_PATH)

    print(f"Ingestion complete! {chunk_count} chunks stored")


if __name__ == "__main__":
    ingest()
