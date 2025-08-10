import requests
from flask import Flask, render_template, request
from bs4 import BeautifulSoup
import nltk
from nltk.util import ngrams
from collections import deque

# --- CONFIGURATION ---
app = Flask(__name__)
# How many top URLs to check from search results
SEARCH_RESULT_COUNT = 30
# The n-gram size (3 means we compare sequences of 3 words)
NGRAM_SIZE = 3
# The similarity threshold (e.g., 0.2 means 20% match)
PLAGIARISM_THRESHOLD = 0.2

# --- HELPER FUNCTIONS ---

def get_search_urls(query):
    """Searches the web and returns a list of top URLs."""
    urls = []
    # Use DuckDuckGo which is generally scraper-friendly
    search_url = "https://html.duckduckgo.com/html/"
    params = {"q": query}
    headers = {"User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/58.0.3029.110 Safari/537.36"}
    
    try:
        response = requests.post(search_url, data=params, headers=headers)
        response.raise_for_status()
        soup = BeautifulSoup(response.text, 'html.parser')
        
        # Extract links from search results
        for link in soup.find_all('a', class_='result__a', href=True):
            urls.append(link['href'])
            if len(urls) >= SEARCH_RESULT_COUNT:
                break
    except requests.exceptions.RequestException as e:
        print(f"Error during search: {e}")
    return urls

def fetch_url_content(url):
    """Fetches and cleans the visible text content from a URL."""
    try:
        response = requests.get(url, timeout=5)
        response.raise_for_status()
        soup = BeautifulSoup(response.content, 'html.parser')
        
        # Remove script and style elements
        for script_or_style in soup(["script", "style"]):
            script_or_style.decompose()
        
        # Get text and clean it up
        text = ' '.join(soup.stripped_strings)
        return text
    except requests.exceptions.RequestException as e:
        print(f"Error fetching {url}: {e}")
        return None

def calculate_jaccard_similarity(text1, text2):
    """Calculates similarity using Jaccard Index on n-grams."""
    # Tokenize and create n-grams
    tokens1 = nltk.word_tokenize(text1.lower())
    tokens2 = nltk.word_tokenize(text2.lower())
    
    ngrams1 = set(ngrams(tokens1, NGRAM_SIZE))
    ngrams2 = set(ngrams(tokens2, NGRAM_SIZE))
    
    if not ngrams1 or not ngrams2:
        return 0.0

    # Calculate Jaccard similarity
    intersection = ngrams1.intersection(ngrams2)
    union = ngrams1.union(ngrams2)
    
    similarity = len(intersection) / len(union)
    return similarity

# --- FLASK ROUTES ---

@app.route('/', methods=['GET', 'POST'])
def index():
    if request.method == 'POST':
        user_text = request.form.get('text_to_check')
        if not user_text or len(user_text.split()) < NGRAM_SIZE:
            return render_template('index.html', error="Please enter sufficient text to check.")

        # 1. Get sentences from user text to use as search queries
        sentences = nltk.sent_tokenize(user_text)
        # Use a deque to avoid re-checking the same URLs
        urls_to_check = deque()
        checked_urls = set()

        # 2. Search using a few key sentences (e.g., first, middle, last)
        search_queries = list(set([sentences[0], sentences[len(sentences)//2], sentences[-1]]))
        for query in search_queries:
            found_urls = get_search_urls(query)
            for url in found_urls:
                if url not in checked_urls:
                    urls_to_check.append(url)
                    checked_urls.add(url)
        
        # 3. Check each unique URL for plagiarism
        plagiarism_results = []
        for url in list(urls_to_check):
            web_content = fetch_url_content(url)
            if web_content:
                similarity_score = calculate_jaccard_similarity(user_text, web_content)
                if similarity_score > PLAGIARISM_THRESHOLD:
                    plagiarism_results.append({
                        "url": url,
                        "score": round(similarity_score * 100)
                    })
        
        # Sort results by score, descending
        plagiarism_results.sort(key=lambda x: x['score'], reverse=True)
        
        return render_template('results.html', results=plagiarism_results)

    return render_template('index.html')

if __name__ == '__main__':
    app.run(debug=True)
