import os
from typing import TypedDict

from dotenv import load_dotenv
from google import genai
from langsmith import traceable
from psycopg.rows import dict_row
from psycopg_pool import ConnectionPool

load_dotenv()

api_key = os.getenv("GOOGLE_API_KEY")

if not api_key:
    raise ValueError("❌ GOOGLE_API_KEY is not defined in your .env file!")

client = genai.Client(api_key=api_key)


class Metadata(TypedDict):
    source: str
    chunkIndex: int


class SearchResult(TypedDict):
    content: str
    metadata: Metadata
    distance: float


DB_CONFIG = {
    "host": "localhost",
    "port": os.getenv("DB_PORT"),
    "user": os.getenv("DB_USER"),
    "password": os.getenv("DB_PASSWORD"),
    "dbname": os.getenv("DB_NAME"),
    "row_factory": dict_row,
}


@traceable(name="Gemini 2.5 Flash - Grounded Answer", run_type="llm")
def _grounded_answer(context: str, user_input: str) -> str:
    """Generate an answer using ONLY the retrieved context."""
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
        f"User Question: {user_input}\n"
        "Answer:"
    )
    response = client.models.generate_content(
        model="gemini-2.5-flash", contents=prompt, config={"temperature": 0.1}
    )
    return response.text or ""


@traceable(name="Similarity Search", run_type="retriever")
def _similarity_search(
    pool: ConnectionPool, vector: list[float], limit: int = 3
) -> list[SearchResult]:
    """Find the `limit` most similar chunks using cosine distance (<=>)."""
    with pool.connection() as conn:
        rows = conn.execute(
            """
            SELECT content, metadata, embedding <=> %s::vector AS distance
            FROM document_vectors
            ORDER BY distance 
            LIMIT %s
            """,
            (str(vector), limit),
        ).fetchall()
        return rows


@traceable(name="Embed User Questions", run_type="embedding")
def _embed_questions(questions: list[str]) -> list[list[float]]:
    """Embed multiple query variations in one API call."""
    result = client.models.embed_content(model="gemini-embedding-2", contents=questions)

    return [
        embedding.values
        for embedding in result.embeddings
        if isinstance(embedding.values, list)
    ]


@traceable(name="Gemini 2.5 Flash - Query Translation", run_type="llm")
def _translate_query(user_input: str) -> str:
    """Ask Gemini to produce 2 alternative phrasings of the user's question."""
    prompt = f"""\
You are an AI assistant. Your job is to look at a user's search query for a document and generate exactly 2 alternative variations of the query that mean the same thing but use different corporate or structural terms.
Format your output as a simple comma-separated list.
Input: {user_input}
Output:"""

    response = client.models.generate_content(
        model="gemini-2.5-flash", contents=prompt, config={"temperature": 0.1}
    )

    return response.text or ""


@traceable(name="PDF RAG Pipeline")
def run_rag_pipeline(pool: ConnectionPool, user_input: str) -> str:
    """Full RAG: translate -> embed -> search -> dedup -> ground -> answer."""
    print(f'\nTranslating query: "{user_input}"...')

    # 1. Multi-query translation: 1 question -> 3 search variations

    translation = _translate_query(user_input)

    queries_to_search = [
        q.strip()
        for q in [user_input, *(translation.split(",") if translation else [])]
        if q.strip()
    ]

    print("--- DEBUG: Query Variations ---", queries_to_search)

    # 2. Embed all query variations
    query_vectors = _embed_questions(queries_to_search)

    # 3. Similarity search for each variation
    all_search_results = [
        _similarity_search(pool, vector, 3) for vector in query_vectors
    ]
    print("--- DEBUG: All Search Results ---", len(all_search_results))

    # 4. Flatten
    flat_results = []

    for sublist in all_search_results:
        for item in sublist:
            flat_results.append(item)

    print("--- DEBUG: Flat Results ---", len(flat_results))

    # 5. Deduplicate

    unique_search_results = {
        search_result["content"]: search_result for search_result in flat_results
    }

    print("--- DEBUG: Relevant Search Results ---", len(unique_search_results))

    # 6. Build context and generate grounded answer
    context = "\n---\n".join(
        search_result["content"] for search_result in unique_search_results.values()
    )
    answer = _grounded_answer(context, user_input)

    print(f"\nAI: {answer}\n")
    return answer


def chat() -> None:
    """Interactive chat loop."""
    # Connection pool: chat makes MANY queries over time, so reuse connections

    with ConnectionPool(kwargs=DB_CONFIG, min_size=1, max_size=4) as pool:
        print("Connected to database. Ready to query!\n")

        while True:
            user_input = input("Ask a question about the PDF (or type 'exit'): ")

            if user_input.strip().lower() == "exit":
                print("Exiting chat session...")
                break
            if not user_input.strip():
                continue

            try:
                run_rag_pipeline(pool, user_input)
            except Exception as e:
                print(f"Error during search or generation: {e}")


if __name__ == "__main__":
    chat()
