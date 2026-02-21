# 2) # fetches policy text and saves as PDF

import requests
from bs4 import BeautifulSoup
from fpdf import FPDF
import re
import os
from scraper import get_soup, user_select_section, find_policy_links
import unicodedata

BASE_URL = "https://www.udst.edu.qa"
INDEX_URL = f"{BASE_URL}/about-udst/institutional-excellence-ie/udst-policies-and-procedures"

pdf_dir = "policy_pdfs"
txt_dir = "policy_txts"
os.makedirs(pdf_dir, exist_ok=True)
os.makedirs(txt_dir, exist_ok=True)  # this line was missing

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

        policy_content = []

        # Extract headings (h2, h3) - important for structured answers
        for heading in soup.find_all(['h2', 'h3']):
            policy_content.append("\n📌 " + heading.get_text(strip=True) + "\n")

        # Extract paragraphs
        for p in soup.find_all('p'):
            policy_content.append(p.get_text(strip=True))

        # Extract list items (bullet points)
        for ul in soup.find_all('ul'):
            for li in ul.find_all('li'):
                policy_content.append("• " + li.get_text(strip=True))  # Bullet point formatting

        # Extract table data (if applicable)
        for table in soup.find_all('table'):
            for row in table.find_all('tr'):
                row_data = [td.get_text(strip=True) for td in row.find_all('td')]
                policy_content.append(" | ".join(row_data))

        policy_text = "\n".join(policy_content)
        return clean_text(policy_text)
    except requests.RequestException as e:
        print(f"Error fetching {url}: {e}")
        return None

def save_text_as_pdf(policy_name, text):
    """Saves extracted text as a PDF, handling Unicode encoding errors."""
    if text is None:
        print(f"Skipping {policy_name} - no content fetched")
        return None

    pdf_path = os.path.join(pdf_dir, f"{policy_name.replace(' ', '_')}.pdf")

    pdf = FPDF()
    pdf.set_auto_page_break(auto=True, margin=10)
    pdf.add_page()
    pdf.set_font("Helvetica", size=11)

    # Add policy title (with better Unicode handling)
    title = clean_text(policy_name)
    pdf.cell(200, 10, title, ln=True, align='C')
    pdf.ln(10)

    for line in text.split("\n"):
        cleaned_line = clean_text(line)  # Apply Unicode filtering
        pdf.multi_cell(190, 8, cleaned_line)  # Keep structured format
        pdf.ln(2)

    pdf.output(pdf_path)
    return pdf_path

# New
def fetch_and_save_policy_text(url: str, policy_name: str) -> str | None:
    """Fetches policy page and saves clean structured text to a .txt file."""
    try:
        response = requests.get(url, timeout=10)
        response.raise_for_status()
        soup = BeautifulSoup(response.content, "html.parser")

        # Remove noise
        for tag in soup.find_all(["nav", "footer", "header", "script", "style", "aside"]):
            tag.decompose()

        main = soup.find("main") or soup.find("article") or soup.body
        blocks = []

        for element in main.descendants:
            if not hasattr(element, "get_text"):
                continue
            text = element.get_text(separator=" ", strip=True)
            if not text or len(text) < 10:
                continue
            if element.name in ("h1", "h2", "h3", "h4"):
                blocks.append(f"\n## {text}\n")
            elif element.name in ("p", "li"):
                blocks.append(text)
            elif element.name in ("td", "th"):
                blocks.append(text)

        # Deduplicate
        seen, clean_blocks = set(), []
        for b in blocks:
            if b not in seen:
                seen.add(b)
                clean_blocks.append(b)

        full_text = "\n".join(clean_blocks)

        # Save as .txt
        txt_path = os.path.join(txt_dir, policy_name.replace(" ", "_") + ".txt")
        with open(txt_path, "w", encoding="utf-8") as f:
            f.write(full_text)

        return full_text

    except requests.RequestException as e:
        print(f"  [ERROR] {url}: {e}")
        return None



# if __name__ == "__main__":
#     soup = get_soup(INDEX_URL)
#     section = user_select_section(soup)
#     policy_links = find_policy_links(soup, section)

#     print("-" * 75)
#     for policy_name, url in policy_links.items():
#         print(f"Downloading {policy_name}...")
#         policy_text = fetch_policy_text(url)
        
#         if policy_text:
#             pdf_path = save_text_as_pdf(policy_name, policy_text)
#             if pdf_path:
#                 print(f"Saved PDF: {pdf_path}\n")

if __name__ == "__main__":
    soup = get_soup(INDEX_URL)
    section = user_select_section(soup)
    policy_links = find_policy_links(soup, section)

    for policy_name, url in policy_links.items():
        print(f"Downloading: {policy_name}")
        
        # Save clean .txt (used for chunking)
        text = fetch_and_save_policy_text(url, policy_name)
        
        # Save PDF (for human reading)
        if text:
            save_text_as_pdf(policy_name, text)
            print(f"  ✅ Saved txt + pdf\n")
