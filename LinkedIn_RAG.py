#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import openai
import time
import numpy as np
from pymongo import MongoClient
from sentence_transformers import SentenceTransformer
from zenml.steps import step
from zenml.pipelines import pipeline
from dotenv import load_dotenv
import os
import re


# Load environment variables
load_dotenv()

MONGO_URI = os.getenv("MONGO_URI")
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
openai.api_key = OPENAI_API_KEY

# Load SentenceTransformer Model for embeddings
embedding_model = SentenceTransformer("all-MiniLM-L6-v2")

# ------------------------------
# STEP 1: EXTRACT LINKEDIN POSTS
# ------------------------------
@step
def extract_posts() -> list:
    """Extracts LinkedIn job posts using Selenium."""
    # Placeholder for actual web scraping code to get LinkedIn posts
    posts = [
        "Hiring for Data Scientist position. Apply now!",
        "Software engineer wanted for a top tech company.",
        "We are recruiting for an AI/ML role in our startup.",
        "Join our team! Positions in Data Science and AI."
    ]
    print(f"[Step 1: Extracted {len(posts)} LinkedIn posts]\n", posts)
    return posts


# ------------------------------
# STEP 2: CLEAN TEXT
# ------------------------------
@step
def clean_text(posts: list) -> list:
    """Cleans and filters LinkedIn job posts."""
    job_keywords = ["hiring", "job", "position", "opening", "opportunity", "career", "recruiting"]
    cleaned_posts = []

    for post in posts:
        if any(word in post.lower() for word in job_keywords):
            post = re.sub(r"\b\d+\b", "", post)  # Remove numbers
            post = re.sub(r"[^\w\s,.!?\'\"-]", "", post)  # Remove special characters
            post = re.sub(r"\b(feed|like|comment|followers?|views?|shares?)\b", "", post, flags=re.IGNORECASE)  # Remove engagement terms
            post = re.sub(r"\s+", " ", post).strip()
            cleaned_posts.append(post)

    print(f"[Step 2: Cleaned {len(cleaned_posts)} posts]\n", cleaned_posts)
    return cleaned_posts


# ------------------------------
# STEP 3: CATEGORIZE POSTS
# ------------------------------
@step
def categorize_posts(posts: list) -> list:
    """Labels LinkedIn job posts into categories."""
    categories = {
        "AI/ML": ["machine learning", "AI", "artificial intelligence", "ML", "deep learning"],
        "NLP": ["natural language processing", "LLM", "transformers", "chatbot"],
        "Data Science": ["data scientist", "analytics", "big data"],
        "Software Engineering": ["software engineer", "developer", "backend", "frontend"],
        "Human Resources": ["HR", "talent acquisition", "recruiter"]
    }

    labeled_posts = []
    for post in posts:
        matched_categories = [cat for cat, keywords in categories.items() if any(kw in post.lower() for kw in keywords)]
        labeled_posts.append({"text": post, "categories": matched_categories if matched_categories else ["Other"]})

    print(f"[Step 3: Categorized Posts]\n", labeled_posts)
    return labeled_posts


# ------------------------------
# STEP 4: CHUNK & EMBED POSTS
# ------------------------------
@step
def chunk_and_embed(posts: list) -> list:
    """Chunks, embeds posts, and stores embeddings in ChromaDB."""
    # ChromaDB setup (dummy here for example purposes)
    # collection = chromadb.PersistentClient(path="./chromadb").get_or_create_collection("linkedin_posts")

    for i, post in enumerate(posts):
        text_chunks = [post["text"][j:j+300] for j in range(0, len(post["text"]), 300)]
        embeddings = embedding_model.encode(text_chunks)
        for k, chunk in enumerate(text_chunks):
            # Saving to ChromaDB (Dummy implementation here)
            print(f"Stored chunk {k} of post {i}: {chunk}")
    
    print(f"[Step 4: Chunked & Embedded Posts in ChromaDB]\n")
    return posts


# ------------------------------
# STEP 5: RETRIEVE SIMILAR POSTS
# ------------------------------
@step
def retrieve_similar_posts(query: str) -> list:
    """Retrieves relevant posts from ChromaDB."""
    query_embedding = embedding_model.encode([query]).tolist()[0]
    
    # Dummy retrieval from "ChromaDB"
    results = [
        {"text": "We are recruiting for an AI/ML role in our startup.", "categories": ["AI/ML"]},
        {"text": "Join our team! Positions in Data Science and AI.", "categories": ["AI/ML", "Data Science"]}
    ]
    
    print(f"[Step 5: Retrieved Similar Posts for Query: {query}]\n", results)
    return results


# ------------------------------
# STEP 6: GENERATE RESPONSE WITH RAG
# ------------------------------
@step
def generate_response(query: str, retrieved_docs: list) -> str:
    """Generates response using OpenAI API with retrieved context."""
    retrieved_context = " ".join([doc["text"] for doc in retrieved_docs])

    pre_rag_prompt = query
    post_rag_prompt = f"Context: {retrieved_context}\n\nUser Query: {query}\n\nAnswer using the provided context."

    print(f"[Pre-RAG Query]: {pre_rag_prompt}")
    print(f"[Post-RAG Query]: {post_rag_prompt}\n")

    response = openai.chat.completions.create(model="gpt-4",
    messages=[{"role": "system", "content": "You are an AI job assistant."},
              {"role": "user", "content": post_rag_prompt}])

    generated_response = response.choices[0].message.content
    print(f"[Step 6: Generated Response]\n", generated_response)
    return generated_response


# ------------------------------
# ZENML PIPELINE: LINKEDIN RAG
# ------------------------------
@pipeline
def linkedin_rag_pipeline():
    """Complete LinkedIn RAG Pipeline."""
    raw_posts = extract_posts()
    cleaned_posts = clean_text(raw_posts)
    categorized_posts = categorize_posts(cleaned_posts)
    chunk_and_embed(categorized_posts)


# ------------------------------
# RUN THE PIPELINE & INTERACTIVE QUERIES
# ------------------------------
if __name__ == "__main__":
    linkedin_rag_pipeline()

    while True:
        user_query = input("\nEnter your question (or type 'exit' to quit): ").strip()
        if user_query.lower() == "exit":
            print("Exiting pipeline.")
            break

        retrieved_docs = retrieve_similar_posts(user_query)
        response = generate_response(user_query, retrieved_docs)
        print("\nFinal Response:\n", response)

