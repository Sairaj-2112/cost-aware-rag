import pymupdf


def load_pdf(pdf_path):

    pdf = pymupdf.open(pdf_path)

    pages = []

    for page_number, page in enumerate(pdf, start=1):

        text = page.get_text().strip()

        pages.append({
            "page": page_number,
            "text": text
        })

    pdf.close()

    return pages