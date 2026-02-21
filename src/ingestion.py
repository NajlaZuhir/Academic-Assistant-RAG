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
os.makedirs(pdf_dir, exist_ok=True)

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

if __name__ == "__main__":
    soup = get_soup(INDEX_URL)
    section = user_select_section(soup)
    policy_links = find_policy_links(soup, section)

    print("-" * 75)
    for policy_name, url in policy_links.items():
        print(f"Downloading {policy_name}...")
        policy_text = fetch_policy_text(url)
        
        if policy_text:
            pdf_path = save_text_as_pdf(policy_name, policy_text)
            if pdf_path:
                print(f"Saved PDF: {pdf_path}\n")

