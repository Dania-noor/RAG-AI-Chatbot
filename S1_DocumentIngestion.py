import os
import pandas as pd

from langchain_community.document_loaders import (
    PyPDFLoader,
    Docx2txtLoader,
    TextLoader
)

from langchain_core.documents import Document


def load_documents():

    documents = []

    # =========================
    # PDF FILES
    # =========================

    pdf_path = "data/pdf"

    for file in os.listdir(pdf_path):

        if file.endswith(".pdf"):

            file_path = os.path.join(pdf_path, file)

            loader = PyPDFLoader(file_path)

            documents.extend(loader.load())


    # =========================
    # DOCX FILES
    # =========================

    docx_path = "data/docx"

    for file in os.listdir(docx_path):

        if file.endswith(".docx"):

            file_path = os.path.join(docx_path, file)

            loader = Docx2txtLoader(file_path)

            documents.extend(loader.load())


    # =========================
    # TXT FILES
    # =========================

    txt_path = "data/txt"

    for file in os.listdir(txt_path):

        if file.endswith(".txt"):

            file_path = os.path.join(txt_path, file)

            loader = TextLoader(
                file_path,
                encoding="utf-8"
            )

            documents.extend(loader.load())


    # =========================
    # EXCEL FILES
    # =========================

    excel_path = "data/excel"

    for file in os.listdir(excel_path):

        if file.endswith(".xlsx"):

            file_path = os.path.join(excel_path, file)

            df = pd.read_excel(file_path)

            text = df.to_string(index=False)

            documents.append(
                Document(
                    page_content=text,
                    metadata={"source": file_path}
                )
            )


    # =========================
    # RESULTS
    # =========================

    print("=" * 60)
    print("DOCUMENT INGESTION")
    print("=" * 60)

    print("Total documents/pages loaded:", len(documents))

    if documents:

        print("\nFirst document preview:")
        print(documents[0].page_content[:500])

        print("\nMetadata:")
        print(documents[0].metadata)

    return documents


if __name__ == "__main__":

    load_documents()