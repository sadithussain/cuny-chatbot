# Import required libraries
import requests
import csv
from bs4 import BeautifulSoup
import os

# Website URL and file output path
TARGET_URL = "https://www.ccny.cuny.edu/cpdi/interviewing"
OUTPUT_DIR = "data_demo/ccny/student_tips"
OUTPUT_FILENAME = os.path.join(OUTPUT_DIR, "interviewing.csv")

HEADERS = {
    'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36'
}

def scrape_interviewing():
    print(f"Fetching data from {TARGET_URL}...")

    try:
        response = requests.get(TARGET_URL, headers = HEADERS)
        response.raise_for_status()

        soup = BeautifulSoup(response.content, 'lxml')

        scraped_data = []

        content_area = soup.find('div', class_ = 'body-paragraph')

        if not content_area:
                print("Error: Could not find the main content area with class 'body-paragraph'.")
                return
        
        all_headings = content_area.find_all('h4')

        for heading in all_headings:
            title = heading.get_text(strip = True)
            content_parts = []

            for sibling in heading.find_next_siblings():
                if sibling.name == 'h4':
                    break
                
                if sibling.name in ['p', 'ul']:
                    if sibling.name == 'ul':
                        list_items = sibling.find_all('li')
                        for item in list_items:
                            clean_text = ' '.join(item.get_text(strip = True, separator = ' ').split())
                            content_parts.append(f"- {clean_text}")
                    else:
                        clean_text = ' '.join(sibling.get_text(strip = True, separator = ' ').split())
                        if clean_text:
                            content_parts.append(clean_text)
            
            full_content = "\n".join(content_parts)

            if title and full_content:
                scraped_data.append({
                    'tip_title': title,
                    'tip_text': full_content
                })

        if not scraped_data:
            print("No data was scraped. Please check the HTML structure and selectors.")
            return
        
        print(f"Found {len(scraped_data)} sections. Saving to {OUTPUT_FILENAME}...")

        os.makedirs(OUTPUT_DIR, exist_ok = True)

        with open(OUTPUT_FILENAME, 'w', newline = '', encoding = 'utf-8') as f:
            fieldnames = ['tip_title', 'tip_text']
            writer = csv.DictWriter(f, fieldnames = fieldnames)
            writer.writeheader()
            writer.writerows(scraped_data)

        print("Scraping completed successfully!")

    except requests.exceptions.RequestException as e:
        print(f"Error fetching the URL: {e}")
    except Exception as e:
        print(f"An error occurred: {e}")

if __name__ == "__main__":
    scrape_interviewing()