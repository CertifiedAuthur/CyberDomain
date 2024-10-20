# Project README: PDF Processing and Information Retrieval
### Overview
This project involves the processing of PDF documents to extract text, generate embeddings, and implement an advanced information retrieval system using Pinecone. 
The workflow includes loading PDF files, merging them, extracting text, creating embeddings using OpenAI's models, and employing various techniques for querying the indexed data.
The primary goal is to process multiple PDF files, extract meaningful content, and enable efficient querying through an indexed database.

### Table of Contents
Project Setup
PDF Loading and Merging
Text Extraction
Embeddings Creation
Pinecone Index Configuration
Upserting Embeddings
Information Retrieval Techniques
Conclusion

### Project Setup
Ensure you have the required libraries installed. You can set up the environment using pip

``
pip install -r requirmenets.txt
``

### PDF Loading and Merging
We start by loading PDF files from a specified directory and merging them into a single document. This is done using the following function:

``
def load_pdfs_from_directory(directory_path):
    ...
def merge_pdfs(file_path, output_path):
    ...
``

1. Loading PDFs: The load_pdfs_from_directory function iterates through a directory, opens all PDF files, and stores them for processing.
2. Merging PDFs: The merge_pdfs function creates a new PDF document by appending all loaded PDFs.

#### Example Usage:

``
pdf_files = load_pdfs_from_directory("path/to/pdf/directory")
merge_pdfs("path/to/pdf/directory", "merged_output.pdf")
``

### Text Extraction
After merging, the next step is to extract text from the PDF. The split_pdf_and_extract_text function handles this task:

``
def split_pdf_and_extract_text(input_pdf_path, output_dir):
    ...
``

This function splits the merged PDF into individual pages and extracts text from each page, returning a dictionary with page numbers as keys and extracted text as values.

#### Example Usage:

``
extracted_texts = split_pdf_and_extract_text("merged_output.pdf", "output/directory")
``

### Embeddings Creation
With the extracted texts, we proceed to create embeddings using OpenAI's models. These embeddings will allow us to perform semantic search.

``
embeddings = [embedding.embed_query(text) for text in extracted_texts.values()]
``

### Pinecone Index Configuration
Next, we set up a Pinecone index to store our embeddings. The configuration is handled by the following function:

``
def configure_pinecone_index():
    ...
``

This function checks if the index exists; if not, it creates a new one with the required specifications.

#### Example Usage:

``
configure_pinecone_index()
``

### Upserting Embeddings
Once the embeddings are created, we need to upsert them into the Pinecone index. The following function manages this process

``
def upsert_embeddings_to_pinecone(embeddings, extracted_texts, max_size=4194304):
    ...
``

This function ensures that embeddings and their associated texts are stored efficiently without exceeding size limits.

#### Example Usage:

``
upsert_embeddings_to_pinecone(embeddings, extracted_texts)
``

### Information Retrieval Techniques
To enhance the retrieval process, we implement several techniques:

1. Query Expansion: This technique generates multiple variations of a user’s query to improve search relevance.

``
class QueryExpansion:
    ...
``

2. Self Querying: This helps in extracting necessary metadata from user queries.

``
class SelfQuery:
    ...
``

3. Reranking: This improves the order of search results based on relevance to the query.

``
class Reranker:
    ...
``

#### Example Usage

``
expanded_queries = QueryExpansion.generate_response("What is AI?", 5)
filtered_id = SelfQuery.generate_response("Find user ID")
ranked_passages = Reranker.generate_response("What is AI?", passages, keep_top_k=3)
``

### Conclusion
This project effectively showcases a pipeline from document ingestion to intelligent retrieval. By leveraging PyPDF2, PyMuPDF, pdfplumber, OpenAI embeddings, and Pinecone, it demonstrates the ability to handle complex PDF documents and implement advanced search techniques. This system serves as a robust foundation for applications requiring efficient document processing and retrieval capabilities.

#### Portfolio Presentation
As an AI Engineer, this project highlights my proficiency in:

1. Document processing
2. Natural Language Processing (NLP)
3. Building efficient and scalable retrieval systems


#### License
This project is licensed under the MIT License. See the LICENSE file for details.

MIT License Summary
The MIT License is a permissive free software license. It allows you to:

Use, copy, modify, merge, publish, distribute, sublicense, and/or sell copies of the Software.
Allow others to do the same.
However, the software is provided "as is", without warranty of any kind.

