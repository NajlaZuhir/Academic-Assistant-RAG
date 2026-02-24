# 2) # fetches policy text and saves as JSON

import requests
from bs4 import BeautifulSoup
import json
import re
import os
from scraper import get_soup, user_select_section, find_policy_links
import unicodedata
from datetime import datetime

BASE_URL = "https://www.udst.edu.qa"
INDEX_URL = f"{BASE_URL}/about-udst/institutional-excellence-ie/udst-policies-and-procedures"

json_dir = "policy_jsons"
txt_dir = "policy_txts"
os.makedirs(json_dir, exist_ok=True)
os.makedirs(txt_dir, exist_ok=True)

def clean_text(text):
    """Cleans and normalizes text."""
    text = unicodedata.normalize("NFKD", text)
    text = re.sub(r'\s+', ' ', text).strip()
    return unicodedata.normalize("NFKD", text).encode("latin-1", "ignore").decode("latin-1")

def fetch_policy_text(url):
    """Fetches policy text with structured content."""
    try:
        response = requests.get(url, timeout=10)
        response.raise_for_status()
        soup = BeautifulSoup(response.content, 'html.parser')

        # Extract structured data
        headings = []
        paragraphs = []
        lists = []
        tables = []

        # Extract headings (h2, h3)
        for heading in soup.find_all(['h2', 'h3']):
            headings.append(heading.get_text(strip=True))

        # Extract paragraphs
        for p in soup.find_all('p'):
            paragraphs.append(p.get_text(strip=True))

        # Extract list items (bullet points)
        for ul in soup.find_all('ul'):
            list_items = []
            for li in ul.find_all('li'):
                list_items.append(li.get_text(strip=True))
            if list_items:
                lists.append(list_items)

        # Extract table data (if applicable)
        for table in soup.find_all('table'):
            table_data = []
            for row in table.find_all('tr'):
                row_data = [td.get_text(strip=True) for td in row.find_all(['td', 'th'])]
                table_data.append(row_data)
            if table_data:
                tables.append(table_data)

        return {
            "headings": headings,
            "paragraphs": paragraphs,
            "lists": lists,
            "tables": tables
        }
    except requests.RequestException as e:
        print(f"Error fetching {url}: {e}")
        return None

def save_text_as_json(policy_name, policy_url, structured_data):
    """Saves extracted policy as JSON with metadata and structured content."""
    if structured_data is None:
        print(f"Skipping {policy_name} - no content fetched")
        return None

    json_path = os.path.join(json_dir, f"{policy_name.replace(' ', '_')}.json")

    # Clean all text fields
    cleaned_data = {
        "metadata": {
            "policy_name": clean_text(policy_name),
            "url": policy_url,
            "fetched_date": datetime.now().isoformat(),
            "total_headings": len(structured_data["headings"]),
            "total_paragraphs": len(structured_data["paragraphs"]),
            "total_lists": len(structured_data["lists"]),
            "total_tables": len(structured_data["tables"])
        },
        "content": {
            "headings": [clean_text(h) for h in structured_data["headings"]],
            "paragraphs": [clean_text(p) for p in structured_data["paragraphs"]],
            "lists": [[clean_text(item) for item in lst] for lst in structured_data["lists"]],
            "tables": [[[clean_text(cell) for cell in row] for row in table] for table in structured_data["tables"]]
        }
    }

    with open(json_path, 'w', encoding='utf-8') as f:
        json.dump(cleaned_data, f, indent=2, ensure_ascii=False)

    return json_path


if __name__ == "__main__":
    soup = get_soup(INDEX_URL)
    section = user_select_section(soup)
    policy_links = find_policy_links(soup, section)

    print("-" * 75)
    for policy_name, url in policy_links.items():
        print(f"Downloading {policy_name}...")
        structured_data = fetch_policy_text(url)
        
        if structured_data:
            json_path = save_text_as_json(policy_name, url, structured_data)
            if json_path:
                print(f"Saved JSON: {json_path}\n")

