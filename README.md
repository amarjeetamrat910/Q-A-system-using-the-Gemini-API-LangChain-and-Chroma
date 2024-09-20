# PDF Q&A System using Google Generative AI

This project implements a retrieval-augmented generation (RAG) system for question-answering (Q&A) based on PDF documents. It utilizes PDF parsing, vector store embeddings, and language models to extract and summarize information from large documents. The pipeline integrates Google’s Generative AI model (`gemini-pro`) for accurate, context-aware answers.

## Features

- **PDF Parsing**: Reads and processes PDF documents into manageable text chunks.
- **Vector Store**: Stores document embeddings to enable fast and relevant search results.
- **Generative AI**: Utilizes Google Generative AI for natural language processing and question answering.
- **Customizable Parameters**: You can adjust key parameters like document chunk size, search relevance (`k`), temperature, and `top_p` to fine-tune response generation.

## Prerequisites

To run this project,   following version should be installed:

- **Python 3.8+**
- **Libraries**: Install the required Python packages using:
  ```bash
  pip install pymupdf nltk langchain chromadb
  ```

- **Google API Key**:  need an API key to access Google’s Generative AI. Obtain it from the [Google Cloud Platform](https://console.cloud.google.com/), and set it up in your environment.

## System Workflow

### 1. **Environment Setup**
   The API key is required to authenticate and access the Google Generative AI services. The function `setup_environment(api_key)` sets the key as an environment variable.
   
   ```python
   def setup_environment(api_key):
       os.environ['GOOGLE_API_KEY'] = api_key
   ```

### 2. **PDF Loading and Processing**
   The `load_and_process_pdf` function processes a given PDF by extracting its text content, splitting the text based on specified `split_start` and `split_end` markers, and tokenizing it into sentences. These sentences are further divided into chunks based on a configurable `chunk_size`. Each chunk is wrapped into a `Document` object with metadata for future reference.
   
   ```python
   def load_and_process_pdf(file_path, split_start, split_end, chunk_size=500):
       doc = fitz.open(file_path)
       # Processes the PDF into chunks...
       return documents
   ```

### 3. **Vector Store Setup**
   This part creates a vector store (using ChromaDB) for efficient document search. It embeds the document text into vector space using the `GoogleGenerativeAIEmbeddings` model.

   ```python
   def setup_vectorstore(docs, model_name, persist_dir):
       gemini_embeddings = GoogleGenerativeAIEmbeddings(model=model_name)
       vectorstore = Chroma.from_documents(documents=docs, embedding=gemini_embeddings, persist_directory=persist_dir)
       return vectorstore
   ```

### 4. **Retriever Configuration**
   The retriever fetches the top `k` relevant documents from the vector store based on the query. The parameter `k` controls how many top documents are retrieved to answer the question.

   ```python
   def setup_retriever(vectorstore, k):
       return vectorstore.as_retriever(search_kwargs={"k": k})
   ```

### 5. **LLM Setup**
   The function `setup_llm` initializes the Google Generative AI model (`gemini-pro`) with specific parameters such as `temperature` and `top_p`, which control the model’s creativity and response diversity.

   ```python
   def setup_llm(llm_model_name, temperature, top_p):
       llm = ChatGoogleGenerativeAI(model=llm_model_name, temperature=temperature, top_p=top_p)
       # Prepares the LLM and prompt...
       return llm, llm_prompt
   ```

### 6. **Retrieval-Augmented Generation (RAG) Chain**
   The RAG chain combines document retrieval and generation steps. First, the retriever fetches relevant documents, then a format function concatenates them. Finally, the language model generates an answer based on this contextual input.

   ```python
   def create_and_invoke_rag_chain(retriever, llm, llm_prompt, question):
       rag_chain = (
           {"context": retriever | format_docs, "question": RunnablePassthrough()}
           | llm_prompt
           | llm
           | StrOutputParser()
       )
       return rag_chain.invoke(question)
   ```

## Configuration Parameters

 configuring the system through the `configuration()` function. The key parameters include:

- **API Key**: Required for Google AI access.
- **PDF Path**: Path to the PDF file you want to analyze.
- **split_start / split_end**: Markers to extract specific sections of the PDF.
- **model_name**: The embedding model used to vectorize document content.
- **persist_dir**: Directory for saving and loading the vector store.
- **k**: Number of relevant documents to retrieve for a given question.
- **llm_model_name**: Google Generative AI model to use.
- **temperature**: Controls randomness in generated responses (lower values mean more deterministic outputs).
- **top_p**: Controls the diversity of the generated output (higher values include more diverse responses).
  
Example configuration:
```python
def configuration():
    return {
        "api_key": getpass("Enter your API key:"),
        "pdf_path": "/path/to/your/pdf/document.pdf",
        "split_start": "Start of content",
        "split_end": "End of content",
        "model_name": "models/embedding-001",
        "persist_dir": "./chroma_db",
        "k": 3,
        "llm_model_name": "gemini-pro",
        "temperature": 0.7,
        "top_p": 0.85,
        "question": input("Ask your question: ")
    }
```

## Running the System

1. **Set up API key**: Make sure your API key is set before running the system.
2. **Run the script**:
   ```bash
   python your_script.py
   ```
3. The system will prompt you to:
   - Input your API key.
   - Provide a question to ask based on the loaded PDF.
4. It will return a concise answer after processing the PDF and querying the language model.

## Code Structure

- **setup_environment**: Initializes API credentials.
- **load_and_process_pdf**: Extracts and processes text from a PDF.
- **setup_vectorstore**: Converts the text into a searchable vector store.
- **setup_retriever**: Configures a retriever to search the vector store.
- **setup_llm**: Initializes Google’s AI for generating responses.
- **create_and_invoke_rag_chain**: Executes the retrieval-augmented generation chain for Q&A.

## Customization

- **Document Chunks**: Adjust `chunk_size` to control how much text each chunk contains.
- **Model Parameters**: Tune `temperature` and `top_p` to change the behavior of the language model.
- **Document Search**: Change `k` to adjust how many relevant documents are considered when answering a question.

