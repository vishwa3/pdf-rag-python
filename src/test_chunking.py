from chunking import markdown_aware_chunk

text = """
# Employee Benefits

This document outlines the benefits package for all employees.

| Benefit | Description | Amount |
|---------|-------------|--------|
| Health Insurance | Medical coverage | 50000 |
| Dental | Dental coverage | 10000 |
| Vision | Vision coverage | 5000 |

Please review the above benefits and contact HR with questions.
"""

chunks = markdown_aware_chunk(text, max_size=200, overlap=50)

for i in range(len(chunks)):
    print(f"\n=== Chunk {i + 1} length={len(chunks[i])}")
    print(chunks[i])

# --- Assertions: verify table-aware behavior ---
assert chunks[0].startswith("# Employee Benefits"), (
    "Chunk 1 should start with the heading"
)

# Chunk 2 = prose overlap + table — header is CONTAINED, not at position 0
assert "| Benefit | Description | Amount |" in chunks[1], (
    "Table chunk must contain the header row so it's self-describing"
)

# Chunk 3 = table tail overlap + prose
assert chunks[2].startswith("| Vision"), (
    "Chunk 3 should start with the table tail overlap"
)

print("\n✅ All assertions passed — table-aware chunking works correctly!")
