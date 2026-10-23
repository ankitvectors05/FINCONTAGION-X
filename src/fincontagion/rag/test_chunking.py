
from fincontagion.rag.ingestion import extract_pdf_pages
from fincontagion.rag.chunking import chunk_pages


pdf_name = "rbi_fsr_june_2025.pdf"

pages = extract_pdf_pages(pdf_name)
chunks = chunk_pages(pages)

print("Extracted pages:", len(pages))
print("Generated chunks:", len(chunks))

if chunks:
    print("\nFirst chunk metadata:")
    print({key: value for key, value in chunks[0].items() if key != "text"})

    print("\nFirst chunk preview:")
    print(chunks[0]["text"][:300])

    print("\nMaximum chunk length:", max(len(c["text"]) for c in chunks))
else:
    print("No chunks generated.")
