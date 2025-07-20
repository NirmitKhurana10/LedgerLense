from bs4 import BeautifulSoup

def extract_clean_text_from_xbrl(raw_text):
    soup = BeautifulSoup(raw_text, "html.parser")

    cleaned_text = []

    # Get all text from <p> and <td> tags
    for p in soup.find_all("p"):
        cleaned_text.append(p.get_text(strip=True))

    for td in soup.find_all("td"):
        cleaned_text.append(td.get_text(strip=True))

    return "\n".join(cleaned_text)
