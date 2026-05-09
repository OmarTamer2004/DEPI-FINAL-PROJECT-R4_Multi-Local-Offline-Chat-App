from langchain.text_splitter import (
    RecursiveCharacterTextSplitter
)

from langchain.schema.document import Document

from llm_chains import (
    load_vectordb,
    create_embeddings
)

import pypdfium2


# ---------------- EXTRACT PDF TEXT ----------------
def extract_text_from_pdf(pdf_bytes):

    pdf_file = pypdfium2.PdfDocument(pdf_bytes)

    text = ""

    for page_number in range(len(pdf_file)):

        page = pdf_file.get_page(page_number)

        text_page = page.get_textpage()

        text += (
            text_page.get_text_range()
            + "\n"
        )

    return text


# ---------------- GET PDF TEXTS ----------------
def get_pdf_texts(pdfs_bytes_list):

    return [
        extract_text_from_pdf(
            pdf_bytes.getvalue()
        )
        for pdf_bytes in pdfs_bytes_list
    ]


# ---------------- SPLIT TEXT ----------------
def get_text_chunks(text):

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=150,
        separators=[
            "\n\n",
            "\n",
            ".",
            " "
        ]
    )

    return splitter.split_text(text)


# ---------------- CREATE DOCUMENTS ----------------
def get_document_chunks(text_list):

    documents = []

    for text in text_list:

        chunks = get_text_chunks(text)

        for chunk in chunks:

            documents.append(
                Document(
                    page_content=chunk
                )
            )

    return documents


# ---------------- ADD TO VECTOR DB ----------------
def add_documents_to_db(pdfs_bytes):

    if not pdfs_bytes:
        return

    texts = get_pdf_texts(pdfs_bytes)

    documents = get_document_chunks(texts)

    vector_db = load_vectordb(
        create_embeddings()
    )

    vector_db.add_documents(documents)

    print("Documents Added Successfully.")
