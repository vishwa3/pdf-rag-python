import hashlib
import json
import os
from pathlib import Path

from dotenv import load_dotenv
from google import genai
from langsmith import traceable
from psycopg import connect
from psycopg.rows import dict_row

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
    return embedding_result.embeddings[0].values


def ingest() -> None:
    """Run the full ingestion pipeline."""
    print(f"[DB] Connecting to PostgreSQL on port {os.getenv("DB_PORT")}...")
    with connect(**DB_CONFIG) as conn:
        # 1. Enable vector extension & create tracking + vector tables

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
            """
            CREATE TABLE IF NOT EXISTS document_vectors (
                id SERIAL PRIMARY KEY,
                content TEXT NOT NULL,
                metadata JSONB,
                embedding vector(3072)
            )
            """
        )

        # 2. Read the PDF and compute its SHA-256 hash

        file_bytes = Path(FILE_PATH).read_bytes()
        current_file_hash = hashlib.sha256(file_bytes).hexdigest()

        # 3. Check if we've ingested this file before (and if it changed)

        row = conn.execute(
            "SELECT file_hash FROM ingested_files WHERE file_path = %s", (FILE_PATH,)
        ).fetchone()
        print("DEBUG: Existing file check result:", row)

        if row:
            if row["file_hash"] == current_file_hash:
                # SCENARIO 1: File exists and hasn't changed.
                print(f'[SKIP] "{FILE_PATH}" hasn\'t changed. Database is up to date.')
                return
            else:
                # SCENARIO 2: File exists but changed. Purge old vectors.
                print(f'[UPDATE] "{FILE_PATH}" has changed. Purging old vectors...')
                conn.execute(
                    "DELETE FROM document_vectors WHERE (metadata->>'source') = %s",
                    (FILE_PATH,),
                )
        else:
            # SCENARIO 3: Brand new file.
            print(f'[NEW] "{FILE_PATH}" detected. Preparing first-time ingestion...')

        # 4. Extract text from PDF using Gemini Vision (Multimodal & Table-Aware)
        extracted_text = _extract_pdf_to_markdown(file_bytes)
        print(f"Extracted {len(extracted_text)} characters from PDF.")

        # 5. Split using our custom Table-Aware Markdown Chunker!
        chunks = markdown_aware_chunk(extracted_text, 1000, 200)
        print(f"Split into {len(chunks)} table-aware chunks.")

        # 6. Embed each chunk and store in pgvector
        print("Embedding chunks and saving to PostgreSQL...")

        for i in range(len(chunks)):
            embedding_values = _embed_text(chunks[i])
            conn.execute(
                """
                INSERT INTO document_vectors (content, metadata, embedding)
                VALUES (%s, %s, %s::vector)
                """,
                (
                    chunks[i],
                    json.dumps({"source": FILE_PATH, "chunkIndex": i}),
                    json.dumps(embedding_values),
                ),
            )
        print(f"Successfully embedded and stored {len(chunks)} chunks!")

        # 7. Register/Update the hash in the tracking table

        conn.execute(
            """
            INSERT INTO ingested_files (file_path, file_hash, updated_at)
            VALUES (%s, %s, CURRENT_TIMESTAMP)
            ON CONFLICT (file_path)
            DO UPDATE SET file_hash = EXCLUDED.file_hash, updated_at = CURRENT_TIMESTAMP
            """,
            (FILE_PATH, current_file_hash),
        )

        print("Ingestion complete!")


if __name__ == "__main__":
    ingest()
