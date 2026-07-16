import os
import json
from dotenv import load_dotenv
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain_core.documents import Document

load_dotenv()

# Setup paths
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_DIR = os.path.join(BASE_DIR, "data")
RAW_DATA_DIR = os.path.join(DATA_DIR, "raw")
FAISS_INDEX_DIR = os.path.join(DATA_DIR, "faiss_index")

def load_json_file(filename):
    path = os.path.join(RAW_DATA_DIR, filename)
    if not os.path.exists(path):
        print(f"Warning: Cache file not found at {path}")
        return []
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)

def build_documents():
    """Loads cache, formats records, chunks them, and adds citation headers to each chunk."""
    commits = load_json_file("commits.json")
    issues = load_json_file("issues.json")
    pulls = load_json_file("pulls.json")
    
    raw_docs = []
    
    # 1. Process Commits
    for commit in commits:
        sha = commit.get("sha", "")
        short_sha = sha[:7]
        author = commit.get("author", "unknown")
        date = commit.get("date", "")
        subject = commit.get("subject", "")
        body = commit.get("body", "")
        files = ", ".join(commit.get("files", []))
        
        content = (
            f"[Commit {short_sha}]\n"
            f"SHA: {sha}\n"
            f"Author: {author}\n"
            f"Date: {date}\n"
            f"Subject: {subject}\n"
            f"Body: {body}\n"
            f"Files Changed: {files}"
        )
        
        raw_docs.append({
            "content": content,
            "metadata": {
                "source_type": "commit",
                "source_id": sha,
                "citation": f"[Commit {short_sha}]",
                "url": f"https://github.com/fastapi/fastapi/commit/{sha}"
            }
        })
        
    # 2. Process Issues
    for issue in issues:
        num = issue.get("number")
        title = issue.get("title", "")
        author = issue.get("author", "unknown")
        body = issue.get("body", "") or ""
        url = issue.get("html_url", "")
        
        comments_str = ""
        for c in issue.get("comments", []):
            comments_str += f"- [{c.get('author')}]: {c.get('body')}\n"
            
        content = (
            f"[Issue #{num}]\n"
            f"Title: {title}\n"
            f"Author: {author}\n"
            f"URL: {url}\n"
            f"Description: {body}\n"
        )
        if comments_str:
            content += f"Discussion Comments:\n{comments_str}"
            
        raw_docs.append({
            "content": content,
            "metadata": {
                "source_type": "issue",
                "source_id": str(num),
                "citation": f"[Issue #{num}]",
                "url": url
            }
        })

    # 3. Process Pull Requests (PRs)
    for pr in pulls:
        num = pr.get("number")
        title = pr.get("title", "")
        author = pr.get("author", "unknown")
        body = pr.get("body", "") or ""
        url = pr.get("html_url", "")
        
        comments_str = ""
        for c in pr.get("comments", []):
            comments_str += f"- [{c.get('author')}]: {c.get('body')}\n"
            
        content = (
            f"[PR #{num}]\n"
            f"Title: {title}\n"
            f"Author: {author}\n"
            f"URL: {url}\n"
            f"Description: {body}\n"
        )
        if comments_str:
            content += f"Discussion Comments:\n{comments_str}"
            
        raw_docs.append({
            "content": content,
            "metadata": {
                "source_type": "pr",
                "source_id": str(num),
                "citation": f"[PR #{num}]",
                "url": url
            }
        })

    # Split documents while preserving contextual headers
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=150,
        length_function=len
    )
    
    final_docs = []
    for doc in raw_docs:
        chunks = text_splitter.split_text(doc["content"])
        citation_header = doc["metadata"]["citation"]
        source_type = doc["metadata"]["source_type"]
        
        for idx, chunk in enumerate(chunks):
            # If the chunk doesn't already contain the citation identifier, prepend it
            chunk_content = chunk
            if citation_header not in chunk:
                prefix = f"Context Source: {citation_header} ({source_type})\n"
                chunk_content = prefix + chunk
                
            # Create LangChain Document
            meta = doc["metadata"].copy()
            meta["chunk_index"] = idx
            final_docs.append(Document(page_content=chunk_content, metadata=meta))
            
    print(f"Created {len(final_docs)} total chunks from {len(raw_docs)} source items.")
    return final_docs

def ingest_to_faiss():
    """Generates embeddings and builds/saves the FAISS index."""
    import time
    api_key = os.getenv("GEMINI_API_KEY")
    if not api_key:
        print("ERROR: GEMINI_API_KEY environment variable is not set.")
        print("Please create a .env file and set GEMINI_API_KEY before running ingestion.")
        return False
        
    print("Building documents and chunking...")
    docs = build_documents()
    if not docs:
        print("No documents to index. Please run data fetching first.")
        return False
        
    print("Initializing Gemini embeddings...")
    embeddings = GoogleGenerativeAIEmbeddings(
        model="models/gemini-embedding-2",
        google_api_key=api_key
    )
    
    # We batch document ingestion to avoid free tier rate limits
    # Since FAISS allows incremental addition, we process in small batches.
    batch_size = 20
    total_docs = len(docs)
    
    print(f"Creating FAISS index with batching (size={batch_size}) to respect rate limits...")
    db = None
    
    for idx in range(0, total_docs, batch_size):
        batch = docs[idx:idx+batch_size]
        print(f"Processing batch {idx // batch_size + 1} / {((total_docs - 1) // batch_size) + 1} (Documents {idx} to {min(idx + batch_size, total_docs)})...")
        
        try:
            if db is None:
                db = FAISS.from_documents(batch, embeddings)
            else:
                db.add_documents(batch)
        except Exception as e:
            print(f"Error on batch starting at index {idx}: {e}")
            print("Quota limits probably hit. Retrying in 40 seconds...")
            time.sleep(40)
            # Retry
            if db is None:
                db = FAISS.from_documents(batch, embeddings)
            else:
                db.add_documents(batch)
                
        # Sleep for a duration to avoid rate limits
        if idx + batch_size < total_docs:
            print("Waiting 20 seconds to respect free-tier rate limits...")
            time.sleep(20)
            
    if db:
        print(f"Saving FAISS index locally to {FAISS_INDEX_DIR}...")
        db.save_local(FAISS_INDEX_DIR)
        print("Ingestion complete!")
        return True
    return False

if __name__ == "__main__":
    import sys
    print("Starting vector store ingestion...")
    success = ingest_to_faiss()
    if not success:
        sys.exit(1)
