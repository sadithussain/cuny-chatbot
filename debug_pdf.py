# debug_pdf.py
from langchain_community.document_loaders import PyPDFLoader

# Make sure this path points to your calendar file
PDF_PATH = "data_demo/ccny/academic_calendar/Fall 2025 Academic Calendar.pdf"

print(f"--- Loading and extracting text from: {PDF_PATH} ---")

try:
    loader = PyPDFLoader(PDF_PATH)
    pages = loader.load()

    # Loop through each page and print its raw text content
    for i, page in enumerate(pages):
        print(f"\n=============== PAGE {i + 1} ===============\n")
        print(page.page_content)

    print("\n--- Script finished successfully. ---")

except Exception as e:
    print(f"\n--- An error occurred: {e} ---")