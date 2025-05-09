import requests
import time
import re
import json
import random
from concurrent.futures import ThreadPoolExecutor
from tqdm import tqdm  # For progress bar


def get_category_members(category, limit=1000):
    """
    Gets a list of page titles from a given Wikipedia category (up to limit).
    Uses continuation for large categories.
    """
    URL = "https://en.wikipedia.org/w/api.php"
    params = {
        "action": "query",
        "list": "categorymembers",
        "cmtitle": f"Category:{category}",
        "cmlimit": "500",  # Maximum allowed by API
        "format": "json"
    }

    titles = []
    while True:
        try:
            response = requests.get(URL, params=params)
            response.raise_for_status()  # Raise exception for HTTP errors
            data = response.json()
            pages = data.get("query", {}).get("categorymembers", [])

            # Only add article pages (ns=0)
            for page in pages:
                if page["ns"] == 0:  # ns=0 => article
                    titles.append(page["title"])
                if len(titles) >= limit:
                    random.shuffle(titles)  # Fixed the shuffle operation
                    return titles[:limit]

            if "continue" in data:
                params.update(data["continue"])
                time.sleep(0.1)  # Reduced sleep time
            else:
                break
        except requests.exceptions.RequestException as e:
            print(f"Error fetching category members: {e}")
            time.sleep(1)  # Back off on error
            continue

    random.shuffle(titles)  # Fixed the shuffle operation
    return titles[:limit]


def get_article_text(title):
    """
    Gets clean text of a Wikipedia article by title.
    Includes error handling and retries.
    """
    URL = "https://en.wikipedia.org/w/api.php"
    params = {
        "action": "query",
        "titles": title,
        "prop": "extracts",
        "explaintext": True,
        "format": "json"
    }

    max_retries = 3
    for attempt in range(max_retries):
        try:
            res = requests.get(URL, params=params)
            res.raise_for_status()
            data = res.json()
            page = next(iter(data["query"]["pages"].values()))
            return page.get("extract", "")
        except Exception as e:
            if attempt < max_retries - 1:
                time.sleep(1)  # Exponential backoff
            else:
                return ""  # Return empty string after all retries fail


def clean_text(text):
    """
    Removes excess characters, leaves clean text.
    """
    text = re.sub(r"\n+", "\n", text)
    text = re.sub(r"={2,}.*?={2,}", "", text)  # Remove headers
    return text.strip()


def process_article(title):
    """
    Process a single article - for parallel execution.
    """
    try:
        text = get_article_text(title)
        text = clean_text(text)
        if len(text) > 200:  # Filter very short articles
            return text
        return None
    except Exception as e:
        print(f"Error processing {title}: {e}")
        return None


def download_documents(category="Physics", limit=1000, max_workers=10):
    """
    Downloads documents from Wikipedia from selected category.
    Uses parallel processing for speed.
    """
    print(f"Downloading article list from category: {category}")
    titles = get_category_members(category, limit=limit)
    print(f"Found {len(titles)} titles. Downloading content...")

    documents = []

    # Use ThreadPoolExecutor for parallel downloads
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        # Submit all tasks and track with tqdm for progress
        future_to_title = {executor.submit(process_article, title): title for title in titles}

        for future in tqdm(future_to_title, desc="Downloading articles"):
            title = future_to_title[future]
            try:
                result = future.result()
                if result:
                    documents.append(result)
            except Exception as e:
                print(f"Error with {title}: {e}")

    return documents


if __name__ == "__main__":
    # Parse command-line arguments
    import argparse

    parser = argparse.ArgumentParser(description="Download Wikipedia articles from a category")
    parser.add_argument("--category", default="Living people", help="Wikipedia category to scrape")
    parser.add_argument("--limit", type=int, default=100000, help="Maximum number of articles to download")
    parser.add_argument("--workers", type=int, default=14, help="Number of download workers")
    parser.add_argument("--output", default="wiki_documents.json", help="Output JSON file")
    args = parser.parse_args()

    # Download the documents
    docs = download_documents(category=args.category, limit=args.limit, max_workers=args.workers)

    # Save as JSON
    with open(args.output, "w", encoding="utf-8") as f:
        json.dump(docs, f, ensure_ascii=False, indent=2)

    print(f"\nSaved {len(docs)} documents to {args.output}")