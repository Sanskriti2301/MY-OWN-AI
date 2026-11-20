# Your-OWN-AI-Java — Vector Database from Scratch in Java

A Java-based **Vector Database** with an interactive web dashboard.

The project implements three vector similarity search approaches — **HNSW, KD-Tree, and Brute Force** — alongside multiple distance metrics, document storage, and a Retrieval-Augmented Generation (RAG) pipeline powered by Ollama.

Built as an educational project to understand how vector databases and semantic retrieval systems work internally.

## What This Project Does

| Feature | Description |
|---|---|
| 3 Search Algorithms | HNSW, KD-Tree, and Brute Force search |
| 3 Distance Metrics | Cosine similarity, Euclidean distance, and Manhattan distance |
| 16D Demo Vectors | 20 pre-loaded demo vectors across CS, Math, Food, and Sports categories |
| Vector Search | Find the nearest vectors based on similarity |
| Document Management | Insert, search, list, and delete documents |
| Document Chunking | Split long documents into smaller overlapping chunks |
| Embedding Integration | Generate document embeddings using Ollama |
| RAG Pipeline | Retrieve relevant document chunks and generate answers using a local LLM |
| REST API | Backend endpoints for vector operations, document operations, and benchmarking |
| Web Dashboard | Interactive interface for exploring search, documents, and AI features |

## Technology Stack

- **Language:** Java 17
- **Backend Framework:** Spring Boot
- **Build Tool:** Apache Maven
- **Frontend:** HTML, CSS, JavaScript
- **AI Integration:** Ollama
- **Embedding Model:** nomic-embed-text
- **Language Model:** llama3.2
- **Communication:** REST APIs and HTTP

## How It Works

```text
                 User Query / Document
                          |
                          v
                  Embedding Model
                    (Ollama)
                          |
                          v
                  Vector Representation
                          |
                          v
                 Java Vector Database
                          |
             +------------+------------+
             |            |            |
             v            v            v
         Brute Force    KD-Tree       HNSW
             |            |            |
             +------------+------------+
                          |
                          v
                  Similarity Search
                          |
                          v
                 Relevant Documents
                          |
                          v
                   Language Model
                    (Ollama)
                          |
                          v
                    Generated Answer
```

## Search Algorithms

### 1. Brute Force Search

- Compares the query vector against every stored vector.
- Calculates the distance or similarity for each vector.
- Provides an exact nearest-neighbor baseline.
- Useful for comparing the performance of other algorithms.

### 2. KD-Tree

- Organizes vectors using a tree-based spatial partitioning structure.
- Splits the search space across dimensions.
- Can reduce the number of distance calculations for suitable datasets.
- Performance can degrade as dimensionality increases.

### 3. HNSW (Hierarchical Navigable Small World)

- Uses a multilayer graph structure for approximate nearest-neighbor search.
- Higher layers provide shortcuts through the vector space.
- Lower layers refine the search around promising candidates.
- Designed to provide efficient retrieval for larger vector collections.

## Distance Metrics

| Metric | Description |
|---|---|
| Cosine Similarity | Measures the angle between two vectors |
| Euclidean Distance | Measures straight-line distance |
| Manhattan Distance | Measures the sum of absolute coordinate differences |

## Prerequisites

Install the following before running the application:

1. **Java JDK 17 or later**
2. **Apache Maven**
3. **Git**
4. **Ollama** (required for embedding and RAG functionality)

Ollama is an external application. It is not bundled with this repository.

## Installation and Setup (Windows)

### Step 1 — Install Java

Download JDK 17 or later from:

https://adoptium.net/

Verify the installation:

```powershell
java -version
```

### Step 2 — Install Maven

Download Maven from:

https://maven.apache.org/download.cgi

Add Maven's `bin` directory to your Windows PATH.

Verify:

```powershell
mvn -version
```

### Step 3 — Clone the Repository

```powershell
git clone https://github.com/YOUR_USERNAME/Your-OWN-AI-Java.git
cd Your-OWN-AI-Java
```

Replace `YOUR_USERNAME` with your GitHub username.

### Step 4 — Install Ollama

Download Ollama for Windows:

https://ollama.com/download/windows

After installation, download the required models:

```powershell
ollama pull nomic-embed-text
```

```powershell
ollama pull llama3.2
```

Verify the installed models:

```powershell
ollama list
```

Ollama normally runs locally at:

```text
http://localhost:11434
```

**Note:** Ollama is needed for real document embeddings and AI-generated RAG answers. Vector search and algorithm demonstrations can be tested separately, subject to the application's implemented functionality.

### Step 5 — Run the Java Application

Open PowerShell inside the project directory:

```powershell
mvn spring-boot:run
```

Once Spring Boot starts, open:

```text
http://localhost:8080
```

The application dashboard should load in your browser.

## Using the Application

### Tab 1 — Search

- Explore the pre-loaded demo vectors.
- Select a search algorithm: HNSW, KD-Tree, or Brute Force.
- Select a distance metric.
- Execute similarity searches.
- Compare the performance of different search algorithms.

### Tab 2 — Documents

- Enter a document title.
- Paste document content.
- Insert documents into the system.
- Long documents are divided into smaller chunks.
- With Ollama configured, document chunks can be converted into embeddings and indexed for retrieval.

### Tab 3 — Ask AI

- Insert documents before asking questions.
- Enter a question related to the stored documents.
- The system retrieves relevant document chunks.
- The retrieved context is sent to the language model.
- The language model generates an answer based on the retrieved context.

The embedding and language-generation models must be available through Ollama for this workflow.

## REST API Reference

The Spring Boot backend exposes REST endpoints at:

```text
http://localhost:8080
```

### Vector Database Endpoints

| Method | Endpoint | Description |
|---|---|---|
| GET | `/search` | Search for nearest vectors |
| POST | `/insert` | Insert a vector |
| DELETE | `/delete/{id}` | Delete a vector |
| GET | `/items` | Retrieve stored vectors |
| GET | `/benchmark` | Compare search algorithms |
| GET | `/hnsw-info` | Retrieve HNSW information |
| GET | `/stats` | Retrieve database statistics |

### Document and RAG Endpoints

| Method | Endpoint | Description |
|---|---|---|
| POST | `/doc/insert` | Insert and process a document |
| GET | `/doc/list` | List stored documents |
| DELETE | `/doc/delete/{id}` | Delete a document chunk |
| GET | `/doc/search` | Search document chunks |
| POST | `/doc/ask` | Ask questions using RAG |
| GET | `/status` | Check application and Ollama status |

### Additional Endpoints

The application also exposes:

- `GET /status`
- `GET /stats`

These endpoints provide application status and database statistics.

## Project Structure

```text
Your-OWN-AI-Java/
│
├── src/
│   └── main/
│       ├── java/
│       │   └── com/
│       │       └── yourownai/
│       │           └── YourOwnAiApplication.java
│       │
│       └── resources/
│           └── static/
│               └── index.html
│
├── pom.xml
├── README.md
└── .gitignore
```

### Main Components

The Java backend contains the implementation of:

- VectorItem — represents a stored vector.
- BruteForce — exact nearest-neighbor search.
- KDTree — tree-based vector search.
- HNSW — graph-based approximate nearest-neighbor search.
- VectorDB — manages vector storage and retrieval.
- DocumentDB — manages document chunks and their vectors.
- OllamaClient — communicates with local Ollama models.
- Spring Boot REST Controller — exposes application APIs.

## Running Without Ollama

The application can be used to explore the vector search implementation and its algorithms without installing Ollama, provided the relevant endpoints do not require embedding or generation services.

However, the following functionality requires Ollama and its models:

- Generating embeddings for new documents.
- Semantic retrieval using generated document embeddings.
- Generating AI answers through the RAG pipeline.

## Future Improvements

- Persistent vector storage.
- Larger benchmark datasets.
- Search accuracy evaluation using recall@K.
- Performance comparison across different vector dimensions.
- Improved error handling and API validation.
- Authentication and API security.
- Support for additional embedding models.

## Purpose of the Project

This project demonstrates the internal concepts behind vector databases, nearest-neighbor search, and retrieval-augmented generation.

It provides an opportunity to explore how different search algorithms organize and retrieve high-dimensional vector data, and how vector retrieval can be combined with language models to build document-based AI applications.

## License

MIT License.
