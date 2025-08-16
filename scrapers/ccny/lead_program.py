# Import required libraries
import requests
import csv
from bs4 import BeautifulSoup
import os

# Website URL and file output path
TARGET_URL = "https://www.ccny.cuny.edu/cpdi/cuny-leads-program"
OUTPUT_DIR = "data_demo/ccny/student_tips"
OUTPUT_FILENAME = os.path.join(OUTPUT_DIR, "ccny_leads_tips.csv")

HEADERS = {
    'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36'
}

def scrape_leads_program_info():
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

        headings = content_area.find_all('h3')

        for heading_tag in headings:
            title = heading_tag.get_text(strip = True)

            p_tag = heading_tag.find_next_sibling('p')

            text = p_tag.get_text(strip = True)

            if title and text:
                scraped_data.append({
                    'tip_title': title,
                    'tip_text': text
                })

        if not scraped_data:
            print("No data was scraped. Check your selectors.")
            return
        
        print(f"Saving {len(scraped_data)} tips to {OUTPUT_FILENAME}...")

        os.makedirs(OUTPUT_DIR, exist_ok = True)
        with open(OUTPUT_FILENAME, 'w', newline = '', encoding = 'utf-8') as f:
            writer = csv.DictWriter(f, fieldnames = ['tip_title', 'tip_text'])
            writer.writeheader()
            writer.writerows(scraped_data)

        print("Scraping completed successfully!")

    except requests.exceptions.RequestException as e:
        print(f"Error fetching the URL: {e}")
    except Exception as e:
        print(f"An error occurred: {e}")

if __name__ == "__main__":
    scrape_leads_program_info()