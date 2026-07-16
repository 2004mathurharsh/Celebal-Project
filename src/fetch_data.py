import os
import subprocess
import json
import requests
import time
from dotenv import load_dotenv

load_dotenv()

# Setup paths
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_DIR = os.path.join(BASE_DIR, "data")
RAW_DATA_DIR = os.path.join(DATA_DIR, "raw")
REPO_DIR = os.path.join(DATA_DIR, "fastapi_repo")

# Create directories
os.makedirs(RAW_DATA_DIR, exist_ok=True)

GITHUB_TOKEN = os.getenv("GITHUB_TOKEN")

def get_headers():
    headers = {
        "Accept": "application/vnd.github+json",
        "User-Agent": "PatchContext-RAG-App"
    }
    if GITHUB_TOKEN:
        headers["Authorization"] = f"token {GITHUB_TOKEN}"
    return headers

def clone_fastapi():
    """Clones the FastAPI repository locally using a shallow clone if it doesn't exist."""
    if not os.path.exists(REPO_DIR):
        print(f"Cloning FastAPI repository to {REPO_DIR}...")
        try:
            subprocess.run(
                ["git", "clone", "--depth", "300", "https://github.com/fastapi/fastapi", REPO_DIR],
                check=True
            )
            print("Cloning complete.")
        except subprocess.CalledProcessError as e:
            print(f"Error cloning repository: {e}")
            raise e
    else:
        print("FastAPI repository already cloned locally.")

def extract_commits():
    """Extracts git commit history from the cloned repository."""
    print("Extracting commit history...")
    os.chdir(REPO_DIR)
    
    # Run git log with custom format to parse easily
    # Formats: H=hash, an=author name, ad=author date, s=subject, b=body
    cmd = [
        "git", "log", 
        "-n", "300", 
        "--pretty=format:COMMIT_START%nSHA:%H%nAUTHOR:%an%nDATE:%ad%nSUBJECT:%s%nBODY:%b%nFILES_CHANGED:"
    ]
    try:
        output = subprocess.check_output(cmd, text=True, errors='replace')
    except subprocess.CalledProcessError as e:
        print(f"Error running git log: {e}")
        return []

    commits = []
    current_commit = None
    lines = output.splitlines()
    
    i = 0
    while i < len(lines):
        line = lines[i]
        if line.startswith("COMMIT_START"):
            if current_commit:
                commits.append(current_commit)
            current_commit = {"sha": "", "author": "", "date": "", "subject": "", "body": "", "files": []}
            i += 1
            continue
        
        if not current_commit:
            i += 1
            continue
            
        if line.startswith("SHA:"):
            current_commit["sha"] = line[4:].strip()
        elif line.startswith("AUTHOR:"):
            current_commit["author"] = line[7:].strip()
        elif line.startswith("DATE:"):
            current_commit["date"] = line[5:].strip()
        elif line.startswith("SUBJECT:"):
            current_commit["subject"] = line[8:].strip()
        elif line.startswith("BODY:"):
            # Accumulate body until we hit FILES_CHANGED:
            body_lines = []
            i += 1
            while i < len(lines) and not lines[i].startswith("FILES_CHANGED:"):
                body_lines.append(lines[i])
                i += 1
            current_commit["body"] = "\n".join(body_lines).strip()
            continue
        elif line.startswith("FILES_CHANGED:"):
            # The files changed are listed after this line until the next COMMIT_START or end of output
            files = []
            i += 1
            while i < len(lines) and not lines[i].startswith("COMMIT_START"):
                file_line = lines[i].strip()
                if file_line:
                    files.append(file_line)
                i += 1
            current_commit["files"] = files
            continue
        i += 1
        
    if current_commit:
        commits.append(current_commit)
        
    # Return to base directory
    os.chdir(BASE_DIR)
    
    # Save commits
    output_path = os.path.join(RAW_DATA_DIR, "commits.json")
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(commits, f, indent=2, ensure_ascii=False)
    print(f"Saved {len(commits)} commits to {output_path}")
    return commits

def fetch_from_api(url, params=None):
    """Utility to fetch data from GitHub API with rate limit safety."""
    headers = get_headers()
    response = requests.get(url, headers=headers, params=params)
    if response.status_code == 403 and "X-RateLimit-Remaining" in response.headers:
        if response.headers.get("X-RateLimit-Remaining") == "0":
            reset_time = int(response.headers.get("X-RateLimit-Reset", time.time()))
            sleep_duration = max(reset_time - time.time() + 5, 10)
            print(f"Rate limited. Sleeping for {sleep_duration:.1f} seconds...")
            time.sleep(sleep_duration)
            response = requests.get(url, headers=headers, params=params)
            
    if response.status_code != 200:
        print(f"Error fetching {url}: {response.status_code} - {response.text}")
        return None
    return response.json()

def fetch_comments(comments_url):
    """Fetches comments for an issue/PR."""
    comments_data = fetch_from_api(comments_url)
    if not comments_data:
        return []
    
    comments = []
    # Limit to top 5 comments to manage tokens and rate limits
    for c in comments_data[:5]:
        comments.append({
            "author": c.get("user", {}).get("login"),
            "body": c.get("body", ""),
            "created_at": c.get("created_at")
        })
    return comments

def fetch_issues_and_prs():
    """Fetches key design-related issues and PRs from GitHub REST API."""
    print("Fetching design-related issues and PRs from GitHub...")
    
    # Search for closed issues containing 'design' or 'why' to capture architectural discussions
    search_queries = [
        "repo:fastapi/fastapi state:closed label:question",
        "repo:fastapi/fastapi state:closed design",
        "repo:fastapi/fastapi state:closed architecture",
        "repo:fastapi/fastapi state:closed deprecate"
    ]
    
    all_issues = {}
    all_pulls = {}
    
    headers = get_headers()
    if not GITHUB_TOKEN:
        print("WARNING: No GITHUB_TOKEN found. Using unauthenticated requests which are strictly rate-limited.")
        print("We will fetch a very small subset of items to avoid blocking.")
        max_results_per_query = 5
    else:
        max_results_per_query = 15
        
    for query in search_queries:
        url = "https://api.github.com/search/issues"
        params = {"q": query, "per_page": max_results_per_query}
        print(f"Executing search: {query}")
        result = fetch_from_api(url, params=params)
        
        if not result or "items" not in result:
            continue
            
        for item in result["items"]:
            number = item["number"]
            # Check if this is a PR or an Issue
            is_pr = "pull_request" in item
            
            # Basic item details
            details = {
                "number": number,
                "title": item.get("title"),
                "author": item.get("user", {}).get("login"),
                "state": item.get("state"),
                "created_at": item.get("created_at"),
                "body": item.get("body", ""),
                "html_url": item.get("html_url")
            }
            
            # Fetch comments
            comments_url = item.get("comments_url")
            if comments_url and item.get("comments", 0) > 0:
                print(f"  Fetching comments for {'PR' if is_pr else 'Issue'} #{number}...")
                details["comments"] = fetch_comments(comments_url)
                time.sleep(1) # Small delay to respect rate limit
            else:
                details["comments"] = []
                
            if is_pr:
                all_pulls[number] = details
            else:
                all_issues[number] = details
                
    # Save files
    issues_path = os.path.join(RAW_DATA_DIR, "issues.json")
    with open(issues_path, "w", encoding="utf-8") as f:
        json.dump(list(all_issues.values()), f, indent=2, ensure_ascii=False)
    print(f"Saved {len(all_issues)} issues to {issues_path}")
    
    pulls_path = os.path.join(RAW_DATA_DIR, "pulls.json")
    with open(pulls_path, "w", encoding="utf-8") as f:
        json.dump(list(all_pulls.values()), f, indent=2, ensure_ascii=False)
    print(f"Saved {len(all_pulls)} pull requests to {pulls_path}")

if __name__ == "__main__":
    import sys
    print("Starting data collection...")
    try:
        clone_fastapi()
        extract_commits()
        fetch_issues_and_prs()
        print("Data collection completed successfully!")
    except Exception as e:
        print(f"Data collection failed: {e}")
        sys.exit(1)
