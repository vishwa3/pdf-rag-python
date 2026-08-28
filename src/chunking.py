import re


def _get_word_snapped_overlap(text: str, overlapSize: int) -> str:
    if not text or len(text) <= overlapSize:
        return text.strip()

    rawStart = len(text) - overlapSize

    # Snap forward to the next space or newline to keep full words

    spaceIndex = text.find(" ", rawStart)

    newlineIndex = text.find("\n", rawStart)

    snapIndex = rawStart
    if spaceIndex != -1 and newlineIndex != -1:
        snapIndex = min(spaceIndex, newlineIndex) + 1
    elif spaceIndex != -1:
        snapIndex = spaceIndex + 1
    elif newlineIndex != -1:
        snapIndex = newlineIndex + 1

    return text[snapIndex:].strip()


def _split_large_table(table_markdown: str, max_chunk_size: int) -> list[str]:
    lines = table_markdown.split("\n")

    header_row = lines[0] if lines else ""

    separator_row = lines[1] if len(lines) > 1 and "---" in lines[1] else None

    header_block = f"{header_row}\n{separator_row}" if separator_row else header_row

    data_rows = lines[2:] if separator_row else lines[1:]

    chunks: list[str] = []

    current_chunk = header_block

    for row in data_rows:
        candidate = f"{current_chunk}\n{row}"

        if len(candidate) <= max_chunk_size:
            current_chunk = candidate
        else:
            if current_chunk != header_block:
                chunks.append(current_chunk.strip())

            current_chunk = f"{header_block}\n{row}"

            if len(current_chunk) > max_chunk_size:
                print(
                    f" Warning: Table row exceeds max_chunk_size ({len(current_chunk)} > {max_chunk_size})"
                    f" Keeping it as one oversized chunk to preserve table integrity."
                )

    if current_chunk.strip() and current_chunk != header_block:
        chunks.append(current_chunk.strip())

    return chunks


def _recursive_chunk_text(
    text: str,
    max_chunk_size: int = 1000,
    overlap: int = 200,
    separators: list[str] | None = None,
):
    if separators is None:
        separators = ["\n\n", "\n", " ", ""]

    final_chunks: list[str] = []
    separator = separators[-1]
    next_separators: list[str] = []

    for i in range(len(separators)):
        sep = separators[i]
        if sep == "" or sep in text:
            separator = sep
            next_separators = separators[i + 1 :]
            break

    splits = text.split(separator) if separator else list(text)

    current_chunk = ""

    for piece in splits:
        candidate = (
            f"{current_chunk}{separator or ''}{piece}" if current_chunk else piece
        )

        if len(candidate) <= max_chunk_size:
            current_chunk = candidate
        else:
            if current_chunk:
                final_chunks.append(current_chunk.strip())

            if len(piece) > max_chunk_size and next_separators:
                sub_chunks = _recursive_chunk_text(
                    piece, max_chunk_size, overlap, next_separators
                )
                final_chunks.extend(sub_chunks)
                current_chunk = ""
            else:
                overlap_start = max(0, len(current_chunk) - overlap)
                overlap_text = current_chunk[overlap_start:]
                current_chunk = f"{overlap_text} {piece}" if overlap_text else piece

    if current_chunk.strip():
        final_chunks.append(current_chunk.strip())

    return [c for c in final_chunks if c]


def markdown_aware_chunk(
    text: str, max_size: int = 1000, overlap: int = 200
) -> list[str]:

    chunks: list[str] = []
    blocks = re.split(r"\n\n+", text)
    current_chunk = ""
    pending_table_overlap = ""

    for block in blocks:
        trimmed_block = block.strip()
        if not trimmed_block:
            continue
        is_table = trimmed_block.startswith("|")

        if is_table:
            prose_overlap = ""
            if current_chunk.strip():
                chunks.append(current_chunk.strip())
                prose_overlap = _get_word_snapped_overlap(current_chunk, overlap)
                current_chunk = ""

            pending_table_overlap = ""

            table_chunks: list[str]

            if len(trimmed_block) > max_size:
                table_chunks = _split_large_table(trimmed_block, max_size)
            else:
                table_chunks = [trimmed_block]

            if prose_overlap and table_chunks and table_chunks[0]:
                table_chunks[0] = f"{prose_overlap}\n\n{table_chunks[0]}"

            chunks.extend(table_chunks)

            last_table_chunk = table_chunks[-1]

            if last_table_chunk:
                table_lines = last_table_chunk.split("\n")
                accumulated_table_tail = ""

                for i in range(len(table_lines) - 1, -1, -1):
                    line = table_lines[i]

                    if (
                        len(f"{line}\n{accumulated_table_tail}") > overlap
                        and accumulated_table_tail
                    ):
                        break

                    accumulated_table_tail = (
                        line + "\n" + accumulated_table_tail
                        if accumulated_table_tail
                        else line
                    )

                pending_table_overlap = accumulated_table_tail.strip()

            continue

        overlap_head = ""

        if not current_chunk and pending_table_overlap:
            overlap_head = pending_table_overlap
            current_chunk = pending_table_overlap
            pending_table_overlap = ""

        candidate = (
            current_chunk + "\n\n" + trimmed_block if current_chunk else trimmed_block
        )

        if len(candidate) <= max_size:
            current_chunk = candidate
        else:
            if current_chunk.strip() and current_chunk.strip() != overlap_head.strip():
                chunks.append(current_chunk.strip())

            overlap_text = _get_word_snapped_overlap(current_chunk, overlap)

            next_candidate = (
                overlap_text + "\n\n" + trimmed_block if overlap_text else trimmed_block
            )

            if len(next_candidate) <= max_size:
                current_chunk = next_candidate
            else:
                sub_chunks = _recursive_chunk_text(trimmed_block, max_size, overlap)

                if overlap_text and sub_chunks and sub_chunks[0]:
                    sub_chunks[0] = overlap_text + "\n\n" + sub_chunks[0]

                chunks.extend(sub_chunks)
                current_chunk = ""

    if current_chunk.strip():
        chunks.append(current_chunk.strip())

    result: list[str] = []

    for chunk in chunks:
        if not chunk.strip():
            continue
        if result and chunk == result[-1]:
            continue
        result.append(chunk)

    return result
