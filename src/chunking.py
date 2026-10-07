def create_chunks(pages, chunk_size=800, overlap=150):

    chunks = []

    current_text = ""
    current_page = None

    for page in pages:

        text = page["text"].strip()

        if not text:
            continue

        paragraphs = text.split("\n")

        for paragraph in paragraphs:

            paragraph = paragraph.strip()

            if not paragraph:
                continue

            # If adding this paragraph makes the chunk too large
            if len(current_text) + len(paragraph) > chunk_size:

                if current_text:
                    chunks.append({
                        "page": current_page,
                        "text": current_text.strip()
                    })

                # Keep some overlap
                current_text = current_text[-overlap:] + " " + paragraph
                current_page = page["page"]

            else:

                current_text += " " + paragraph

                if current_page is None:
                    current_page = page["page"]

    # Add remaining text
    if current_text.strip():

        chunks.append({
            "page": current_page,
            "text": current_text.strip()
        })

    return chunks