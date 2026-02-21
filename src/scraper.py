# 1) # finds policy links from the index page

import requests
from bs4 import BeautifulSoup
import os

BASE_URL = "https://www.udst.edu.qa"
INDEX_URL = f"{BASE_URL}/about-udst/institutional-excellence-ie/udst-policies-and-procedures"

def get_soup(url: str) -> BeautifulSoup:
    response = requests.get(url)
    return BeautifulSoup(response.text, "html.parser")


def find_policy_sections(soup: BeautifulSoup) -> list:
    sections = []
    for i, tag in enumerate(soup.find_all("h3")):
        print(i + 1, tag.get_text(strip=True))
        sections.append(tag)
    return sections


def user_select_section(soup: BeautifulSoup):
    sections = find_policy_sections(soup)
    section_id = int(input("Enter the number of the section: ").strip())
    return sections[section_id - 1]


def find_policy_links(soup: BeautifulSoup, section_heading) -> dict:
    policy_links = {}

    panel_id = section_heading.find("button")["href"].lstrip("#")
    table = soup.find("div", {"id": panel_id}).find("table")

    for row in table.find_all("tr"):
        cells = row.find_all("td")
        if len(cells) < 2:
            continue

        policy_number = cells[1].get_text(strip=True)
        link = cells[0].find("a")

        if link is None:
            continue

        if policy_number.upper().startswith("PL"):
            policy_links[link.get_text(strip=True)] = BASE_URL + link["href"]

    return policy_links



if __name__ == "__main__":
    soup = get_soup(INDEX_URL)
    section = user_select_section(soup)
    policy_links = find_policy_links(soup, section)

    print(f"\n{'Policy Name':<55} {'URL'}")
    print("-" * 75)
    for name, url in policy_links.items():
        print(f"{name:<55} {url}")
    print(f"\nTotal policies found: {len(policy_links)}")