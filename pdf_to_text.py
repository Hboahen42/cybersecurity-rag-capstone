import pdfplumber
import os

def pdf_to_text(pdf_path: str, txt_output_path: str) -> None:
    with pdfplumber.open(pdf_path) as pdf, open(txt_output_path, "w", encoding="utf-8") as txt_file:
        for page in pdf.pages:
            text = page.extract_text()
            if text:
                txt_file.write(text + "\n\n")

if __name__ == "__main__":
    data_dir = "data_sources"

    for filename in os.listdir(data_dir):
        if filename.endswith(".pdf"):
            pdf_path = os.path.join(data_dir, filename)
            txt_filename = filename.rsplit(".", 1)[0] + ".txt"
            txt_path = os.path.join(data_dir, txt_filename)

            print(f"Converting {pdf_path} to {txt_path}")
            pdf_to_text(pdf_path, txt_path)

    print("All PDFs converted to text files.")
