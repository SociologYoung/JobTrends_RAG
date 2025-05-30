# 🔗 LinkedIn Job RAG System

[![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Code style: black](https://img.shields.io/badge/code%20style-black-000000.svg)](https://github.com/psf/black)

> An intelligent Retrieval-Augmented Generation (RAG) system that extracts, processes, and analyzes LinkedIn job postings to provide contextual AI-powered responses about job opportunities.

## 📋 Table of Contents

- [Overview](#overview)
- [Features](#features)
- [Architecture](#architecture)
- [Quick Start](#quick-start)
- [Installation](#installation)
- [Configuration](#configuration)
- [Usage](#usage)
- [API Reference](#api-reference)
- [Contributing](#contributing)
- [License](#license)

## 🎯 Overview

The LinkedIn Job RAG System is an end-to-end pipeline that combines web scraping, natural language processing, and retrieval-augmented generation to create an intelligent job search assistant. By leveraging ChromaDB for vector storage and OpenAI's GPT models, it provides contextually relevant responses to job-related queries.

### Key Benefits

- **Automated Job Discovery**: Continuously extracts fresh job postings from LinkedIn
- **Intelligent Categorization**: Automatically classifies jobs by role, industry, and skills
- **Contextual Responses**: Provides detailed, relevant answers using RAG methodology
- **Scalable Architecture**: Built with ZenML for production-ready orchestration

## ✨ Features

| Feature 	  | Description |
|-----------------|-------------|
| 🕷️ **Web Scraping**  		| Selenium-based extraction prioritizing job posts over general content |
| 🧹 **Data Cleaning** 		| Advanced text preprocessing and filtering for job-relevant content |
| 📊 **Smart Categorization** 	| ML-powered classification across AI/ML, Data Science, SWE, and more |
| 🔍 **Vector Search** 		| ChromaDB integration for semantic similarity matching |
| 🤖 **RAG Integration** 	| OpenAI-powered responses with retrieved context |
| 📈 **Sentiment Analysis** 	| TextBlob-based sentiment scoring for job posts |
| 🗄️ **Data Persistence** 	| MongoDB and CSV export capabilities |
| ⚡ **Orchestration** 		| ZenML pipeline for scheduling and manual triggers |

## 🏗️ Architecture

```
LinkedIn → Selenium Scraper → Text Cleaner → Categorizer
                                    ↓
ChromaDB ← Embedding Generator ← Text Chunker
   ↓
Query Interface → Vector Search → Context Retrieval → OpenAI GPT → Response
```

## 🚀 Quick Start

```bash
# Clone the repository
git clone https://github.com/yourusername/linkedin-job-rag-system.git
cd linkedin-job-rag-system

# Set up environment
python -m venv venv
source venv/bin/activate  # Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt

# Configure environment variables
cp .env.example .env
# Edit .env with your API keys

# Run the pipeline
python linkedin_rag_pipeline.py
```

## 📦 Installation

### Prerequisites

- Python 3.8 or higher
- Google Chrome browser
- MongoDB instance (local or cloud)
- OpenAI API key

### Dependencies

```bash
pip install -r requirements.txt
```

Core packages:
- `selenium` - Web scraping automation
- `openai` - GPT integration
- `sentence-transformers` - Text embeddings
- `chromadb` - Vector database
- `pymongo` - MongoDB integration
- `zenml` - ML pipeline orchestration
- `textblob` - Sentiment analysis

## ⚙️ Configuration

### Environment Variables

Create a `.env` file in the project root:

```env
# Required
OPENAI_API_KEY=your_openai_api_key_here
MONGO_URI=mongodb://localhost:27017/linkedin_jobs

# Optional
CHROME_DRIVER_PATH=/path/to/chromedriver
LOG_LEVEL=INFO
BATCH_SIZE=50
```

### LinkedIn Setup

1. Ensure Chrome is installed and updated
2. The first run will open a Chrome window for LinkedIn login
3. Complete the login process manually
4. The session will be maintained for subsequent runs

## 💡 Usage

### Interactive Mode

```bash
python linkedin_rag_pipeline.py
```

Then ask questions like:
- "What AI/ML jobs are available?"
- "Show me remote data science positions"
- "Find entry-level software engineering roles"

### Programmatic Usage

```python
from linkedin_rag_pipeline import LinkedInRAGPipeline

pipeline = LinkedInRAGPipeline()

# Extract and process jobs
pipeline.run()

# Query the system
response = pipeline.query("Find Python developer jobs")
print(response)
```

### ZenML Orchestration

```bash
# Schedule pipeline
zenml pipeline schedule linkedin_rag_pipeline --cron "0 9 * * *"

# Manual trigger
zenml pipeline run linkedin_rag_pipeline
```

## 📚 API Reference

### Core Classes

#### `LinkedInRAGPipeline`
Main pipeline orchestrator

**Methods:**
- `run()` - Execute the complete pipeline
- `query(question: str)` - Ask questions about jobs
- `get_jobs_by_category(category: str)` - Filter by job type

#### `JobExtractor`
Handles LinkedIn scraping

**Methods:**
- `extract_posts()` - Scrape job postings
- `filter_job_posts()` - Remove non-job content

#### `EmbeddingManager`
Manages vector operations

**Methods:**
- `generate_embeddings()` - Create text embeddings
- `store_in_chromadb()` - Save to vector database
- `similarity_search()` - Find relevant content

## 🔧 Development

### Code Quality

This project uses:
- **Black** for code formatting
- **Flake8** for linting
- **Pytest** for testing

```bash
# Format code
black .

# Run lints
flake8 .

# Run tests
pytest tests/
```

### Project Structure

```
linkedin-job-rag-system/
├── src/
│   ├── extractors/
│   ├── processors/
│   ├── embeddings/
│   └── pipeline/
├── tests/
├── docs/
├── requirements.txt
├── .env.example
├── .gitignore
└── README.md
```

### Reporting Issues

When reporting bugs, please include:
- Clear description of the issue
- Steps to reproduce
- Expected vs actual behavior
- Environment details (OS, Python version, etc.)

For feature requests, describe the problem you're solving and your proposed solution.

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 🙏 Acknowledgments

- [OpenAI](https://openai.com/) for GPT models
- [ChromaDB](https://www.trychroma.com/) for vector database
- [ZenML](https://zenml.io/) for ML orchestration
- [Sentence Transformers](https://www.sbert.net/) for embeddings

---

<div align="center">
  <sub>Built with ❤️ by [Young]</sub>
</div>