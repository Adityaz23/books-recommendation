<div align="center">

# 📚 Semantic Book Recommender

**Discover your next favorite book using AI-powered semantic search, emotional tone filtering, and category-based recommendations — built with LangChain, ChromaDB, HuggingFace Embeddings, and Gradio.**

[![Python](https://img.shields.io/badge/Python-3.10%2B-blue?style=for-the-badge&logo=python)](https://www.python.org/)
[![Gradio](https://img.shields.io/badge/Gradio-UI-orange?style=for-the-badge&logo=gradio)](https://www.gradio.app/)
[![LangChain](https://img.shields.io/badge/LangChain-RAG-green?style=for-the-badge)](https://www.langchain.com/)
[![ChromaDB](https://img.shields.io/badge/ChromaDB-VectorDB-purple?style=for-the-badge)](https://www.trychroma.com/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow?style=for-the-badge)](LICENSE)

![Demo Screenshot](cover-not-found.png)

</div>

---

## ✨ Features

| Feature | Description |
|---|---|
| 🔍 **Semantic Search** | Describe the kind of book you want in plain English — the AI finds the closest matches |
| 🏷️ **Category Filtering** | Filter results by genre/category (Fiction, Science, History, etc.) |
| 🎭 **Emotional Tone Filtering** | Narrow results by mood — Happy, Sad, Suspenseful, Angry, or Surprising |
| 📖 **Google Books Links** | Direct preview links via ISBN for each recommendation |
| 🛒 **Amazon Search Links** | One-click Amazon search for every recommended book |
| 🖼️ **Book Cover Thumbnails** | High-resolution cover art for every result |
| ⚡ **Pre-built Vector DB** | ChromaDB index is pre-computed and ships with the repo for instant startup |

---

## 🗂️ Project Structure

```
books-recommendation/
├── app.py                        # Main Gradio application
├── requirements.txt              # Python dependencies
├── books_cleaned.csv             # Raw cleaned book dataset
├── books_with_categories.csv     # Dataset with category labels
├── books_with_emotions.csv       # Dataset with emotion scores (used at runtime)
├── tagged_description.txt        # Tagged book descriptions for vector indexing
├── chroma_db/                    # Pre-built ChromaDB vector store
│   ├── chroma.sqlite3
│   └── e24a8b8f-.../             # Embedding shard
├── cover-not-found.png           # Fallback cover image
├── data-exploration.ipynb        # EDA notebook
├── sentiment-analysis.ipynb      # Emotion scoring notebook
├── text-classification.ipynb     # Category classification notebook
└── vector-search.ipynb           # Vector DB creation notebook
```

---

## 🚀 Local Setup Guide

### Prerequisites

Make sure you have the following installed before starting:

| Tool | Version | Link |
|---|---|---|
| Python | 3.10 or higher | [python.org](https://www.python.org/downloads/) |
| pip | Latest | Comes with Python |
| Git | Any | [git-scm.com](https://git-scm.com/) |

---

### Step 1 — Clone the Repository

```bash
git clone https://github.com/Adityaz23/books-recommendation.git
cd books-recommendation
```

---

### Step 2 — Create a Virtual Environment

Using a virtual environment keeps your dependencies isolated and avoids conflicts with other Python projects.

#### 🐧 macOS / Linux

```bash
# Create the virtual environment
python3 -m venv venv

# Activate it
source venv/bin/activate
```

#### 🪟 Windows (Command Prompt)

```cmd
:: Create the virtual environment
python -m venv venv

:: Activate it
venv\Scripts\activate
```

#### 🪟 Windows (PowerShell)

```powershell
# Create the virtual environment
python -m venv venv

# Activate it
.\venv\Scripts\Activate.ps1
```

> ✅ You'll know it's active when you see `(venv)` at the start of your terminal prompt.

---

### Step 3 — Install Dependencies

```bash
pip install -r requirements.txt
```

This installs all required packages:

| Package | Purpose |
|---|---|
| `pandas`, `numpy` | Data loading and manipulation |
| `torch` | PyTorch backend for embeddings |
| `gradio` | Web UI framework |
| `python-dotenv` | Load environment variables from `.env` |
| `langchain` + extensions | RAG pipeline and document handling |
| `langchain-chroma` | ChromaDB integration |
| `langchain-huggingface` | HuggingFace embeddings |
| `sentence-transformers` | `all-MiniLM-L6-v2` embedding model |
| `chromadb` | Local vector database |

> ⚠️ **PyTorch Note:** If you want GPU acceleration or are on a specific OS, install PyTorch separately first from [pytorch.org](https://pytorch.org/get-started/locally/) before running `pip install -r requirements.txt`.

---

### Step 4 — Set Up Environment Variables

The app uses a `.env` file to load your API key securely. Create one in the root of the project:

```bash
# In the project root directory
touch .env        # macOS/Linux
# OR on Windows:
type nul > .env
```

Then open `.env` and add the following:

```env
GROQ_API_KEY=your_groq_api_key_here
Or any other LLM API which you want to use.
```

#### 🔑 How to Get a Groq API Key

1. Go to [https://console.groq.com](https://console.groq.com)
2. Sign up or log in with your account
3. Navigate to **API Keys** in the left sidebar
4. Click **Create API Key**
5. Copy the key and paste it into your `.env` file

> ℹ️ The `GROQ_API_KEY` is loaded by the app but the core semantic search runs entirely via **HuggingFace embeddings + ChromaDB** locally — so the app will still work for recommendations even without the Groq key. The key is used for any future LLM-powered features.

---

### Step 5 — (Optional) Rebuild the Vector Database

The repo ships with a **pre-built ChromaDB index** in the `chroma_db/` folder. You can use it directly and skip this step.

If you want to rebuild it from scratch (e.g., after modifying `tagged_description.txt`):

1. Open `app.py` and **uncomment** this block:

```python
# Run this ONCE to build and save the vector database
db_books = Chroma.from_documents(
    documents,
    embedding=embedding_model,
    persist_directory="chroma_db"
)
```

2. **Comment out** the production loading block below it:

```python
# db_books = Chroma(
#     persist_directory="chroma_db",
#     embedding_function=embedding_model
# )
```

3. Run `app.py` once — it will create the index and save it. Then reverse the comments back for normal usage.

---

### Step 6 — Run the Application

```bash
python app.py
```

The Gradio app will launch and be accessible at:

```
http://localhost:7860
```

Or on your local network at:

```
http://0.0.0.0:7860
```

---

## 🎮 How to Use

Once the app is running:

| Step | Action |
|---|---|
| 1️⃣ | Type a description of the kind of book you're looking for in the text box |
| 2️⃣ | (Optional) Select a **Category** from the dropdown to filter by genre |
| 3️⃣ | (Optional) Select an **Emotional Tone** — Happy, Sad, Suspenseful, Angry, or Surprising |
| 4️⃣ | Click **🔍 Find Recommendations** |
| 5️⃣ | Browse the book cards — click **📖 Read Preview** or **🛒 Buy Book** on any result |

**Example queries to try:**
- `"A mystery thriller set in Victorian London"`
- `"Uplifting stories about human resilience and hope"`
- `"Dark fantasy with complex world-building and magic systems"`
- `"A coming-of-age story about loss and identity"`

---

## 🧠 How It Works

```
User Query
    │
    ▼
HuggingFace Embeddings          ← sentence-transformers/all-MiniLM-L6-v2
(all-MiniLM-L6-v2)
    │
    ▼
ChromaDB Vector Search          ← Top 50 semantically similar books
    │
    ▼
Category Filter                 ← Optional genre filter
    │
    ▼
Emotion Score Re-ranking        ← Sort by joy / sadness / fear / surprise / anger
    │
    ▼
Top 16 Results → Gradio UI      ← Book cards with cover, description & links
```

---

## 📓 Notebooks

The `notebooks/` files walk through the full data pipeline:

| Notebook | Description |
|---|---|
| `data-exploration.ipynb` | EDA of the raw book dataset |
| `text-classification.ipynb` | Classifying books into simplified categories |
| `sentiment-analysis.ipynb` | Scoring books by emotional tone using NLP |
| `vector-search.ipynb` | Building and querying the ChromaDB vector index |

---

## 🛠️ Troubleshooting

| Problem | Fix |
|---|---|
| `ModuleNotFoundError` | Make sure your venv is active and you've run `pip install -r requirements.txt` |
| `No API Key found` printed on startup | Check your `.env` file exists in the root folder and has `GROQ_API_KEY=...` |
| Port 7860 already in use | Change `server_port=7860` in `app.py` to any free port |
| Slow first startup | The HuggingFace embedding model downloads on first use — this is normal |
| Empty recommendations | Try a broader query or set Category and Tone both to "All" |

---

## 🤝 Contributing

Contributions are welcome! To contribute:

1. Fork the repo
2. Create a feature branch: `git checkout -b feature/your-feature`
3. Commit your changes: `git commit -m 'Add your feature'`
4. Push to the branch: `git push origin feature/your-feature`
5. Open a Pull Request

---

## 📄 License

This project is licensed under the **MIT License** — see the [LICENSE](LICENSE) file for details.

---

<div align="center">

Made with ❤️ by [Adityaz23](https://github.com/Adityaz23)

⭐ Star this repo if you found it useful!

</div>
