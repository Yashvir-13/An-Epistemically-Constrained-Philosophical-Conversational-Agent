import os
import re
import json
from langchain.schema import Document

# --- Configuration ---
RAW_DATA_DIR = "Documents/Arthur Schopenhauer"
OUTPUT_FILE = "structured_documents.json"

# --- Document Cleaning Function (Your original code, works well!) ---
def clean_document(raw_text: str) -> str:
    """Removes Project Gutenberg boilerplate and other non-content text."""
    # Find the main content between START and END markers
    start_match = re.search(r"\*\*\* START OF (THE|THIS) PROJECT GUTENBERG EBOOK .* \*\*\*", raw_text)
    end_match = re.search(r"\*\*\* END OF (THE|THIS) PROJECT GUTENBERG EBOOK .* \*\*\*", raw_text)

    if start_match and end_match:
        raw_text = raw_text[start_match.end():end_match.start()]
    elif start_match:
        raw_text = raw_text[start_match.end():]
    elif end_match:
        raw_text = raw_text[:end_match.start()]
    
    # Remove Gutenberg license/reminder sentences if still left
    raw_text = re.sub(r"Project Gutenberg.*?(www\.gutenberg\.org|License)", '', raw_text, flags=re.DOTALL | re.IGNORECASE)

    # Remove translator/editor notes, preface, and table of contents
    raw_text = re.sub(r"(?i)(translator's note|translator's preface|translators' preface|preface|editor's note|transcriber's note|index|table of contents|contents)\b.*?(?=\n\n\w)", '', raw_text, flags=re.DOTALL)

    # Remove illustrations, excessive line breaks
    raw_text = re.sub(r'\[Illustration:.*?\]', '', raw_text)
    raw_text = re.sub(r'\n{3,}', '\n\n', raw_text)
    raw_text = raw_text.replace('\r\n', '\n').strip()
    
    return raw_text

# --- New Structural Parsing Function ---
def parse_schopenhauer_text(text: str, source_file: str) -> list[Document]:
    """
    Parses the full text of a Schopenhauer volume, creating documents
    for each chapter/section with appropriate metadata.
    
    Logic:
    - Uses Regex to identify Roman Numeral headings (BOOK I, CHAPTER IV, etc.).
    - Splits text into semantic chunks (Sections).
    - Ignores small/empty chunks (<200 chars).
    """
    """
    Parses the full text of a Schopenhauer volume, creating documents
    for each chapter/section with appropriate metadata.
    """
    # This regex is designed to find major structural breaks.
    # It looks for "BOOK" or "CHAPTER" followed by Roman numerals, or a section "§".
    # You may need to adjust this pattern if your text files have a different format.
    pattern = r"(BOOK [IVXLC]+\..*|CHAPTER [IVXLC]+\..*|§ \d+\..*)"
    
    # Split the text by these structural headings.
    # The capturing group in the regex ensures the delimiters (headings) are kept.
    parts = re.split(pattern, text)
    
    documents = []
    
    # The output of split is [content_before, delimiter1, content1, delimiter2, content2, ...]
    # We iterate through it in pairs (delimiter, content).
    i = 1
    while i < len(parts):
        heading = parts[i].strip().replace('\n', ' ')
        content = parts[i+1].strip()
        
        # We only care about chunks with substantial content to avoid empty splits.
        if len(content) > 200: 
            metadata = {
                "source": source_file,
                "heading": heading
            }
            doc = Document(page_content=content, metadata=metadata)
            documents.append(doc)
        i += 2
        
    print(f"    -> Parsed {len(documents)} structural sections from {source_file}.")
    return documents

# --- Main Execution Logic ---
if __name__ == "__main__":
    if not os.path.exists(RAW_DATA_DIR):
        print(f"Error: Raw data directory not found at '{RAW_DATA_DIR}'")
        exit()

    all_structural_docs = []
    files = os.listdir(RAW_DATA_DIR)

    print("Starting parsing process...")
    for file in files:
        file_path = os.path.join(RAW_DATA_DIR, file)
        if os.path.isfile(file_path) and file.endswith('.txt'):
            print(f"Processing '{file}'...")
            with open(file_path, 'r', encoding='utf-8') as f:
                raw_text = f.read()
            
            # 1. Clean the document
            cleaned_text = clean_document(raw_text)
            
            # 2. Parse the cleaned text into structural chunks
            parsed_docs = parse_schopenhauer_text(cleaned_text, file)
            all_structural_docs.extend(parsed_docs)

    # Convert Document objects to a JSON-serializable format (list of dicts)
    docs_for_json = [
        {"page_content": doc.page_content, "metadata": doc.metadata}
        for doc in all_structural_docs
    ]

    # Save the structured documents to a single JSON file for the next step
    with open(OUTPUT_FILE, 'w', encoding='utf-8') as f:
        json.dump(docs_for_json, f, indent=2, ensure_ascii=False)

    print(f"\nProcess complete. Found {len(all_structural_docs)} total sections.")
    print(f"All structured documents saved to '{OUTPUT_FILE}'.")
