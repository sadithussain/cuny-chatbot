# Import required libraries
import requests
import csv
from bs4 import BeautifulSoup
import os
import time

TARGET_URL = "https://www.ccny.cuny.edu/cpdi/what-can-i-do-with-this-major"
OUTPUT_DIR = "data_demo/ccny/student_tips"
OUTPUT_FILENAME = os.path.join(OUTPUT_DIR, "ccny_major_details.csv")

HEADERS = {
    'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36'
}

def get_major_links():
    print(f"Fetching list of majors from {TARGET_URL}...")

    try:
        response = requests.get(TARGET_URL, headers = HEADERS)
        response.raise_for_status()

        soup = BeautifulSoup(response.content, 'lxml')

        content_area = soup.find('div', class_ = 'CS_Textblock_Text')
        major_links = content_area.find_all('a')

        links_to_scrape = []
        for link in major_links:
            major_name = link.get_text(strip=True)
            url = link.get('href')
            if major_name and url and "whatcanidowiththismajor.com" in url:
                links_to_scrape.append({'major_name': major_name, 'url': url})
        
        print(f"Found {len(links_to_scrape)} major pages to scrape.")
        return links_to_scrape
    
    except Exception as e:
        print(f"Error fetching the hub page: {e}")
        return []
    
def scrape_all_majors(major_links):
    """Step 2: Visit each major's URL and scrape the detailed information."""
    all_scraped_data = []
    
    for major_info in major_links:
        major_name = major_info['major_name']
        url = major_info['url']
        print(f"\nScraping details for '{major_name}' from {url}...")
        
        try:
            response = requests.get(url, headers=HEADERS)
            response.raise_for_status()
            soup = BeautifulSoup(response.content, 'lxml')

            # The content is organized in accordion items
            accordion_items = soup.find_all('div', class_='wp-block-pb-accordion-item')
            
            for item in accordion_items:
                category = item.find('h2').get_text(strip=True) if item.find('h2') else "General Info"
                
                # Find all sub-headings and their lists
                sub_headings = item.find_all('h3')
                for sub_head in sub_headings:
                    sub_category = sub_head.get_text(strip=True)
                    # Find the list that comes right after the sub-heading
                    ul = sub_head.find_next_sibling('ul')
                    if ul:
                        for li in ul.find_all('li'):
                            all_scraped_data.append({
                                'major': major_name,
                                'category': category,
                                'sub_category': sub_category,
                                'item': li.get_text(strip=True)
                            })
            
            # Be respectful and wait a moment before the next request
            time.sleep(1)

        except Exception as e:
            print(f"  -- Could not scrape {url}. Error: {e}")
            continue
            
    return all_scraped_data

if __name__ == "__main__":
    # First, get the list of all pages we need to visit.
    links = get_major_links()
    
    if links:
        # Then, scrape all of those pages.
        final_data = scrape_all_majors(links)
        
        if final_data:
            # Finally, save all the combined data to a single CSV file.
            print(f"\nSaving a total of {len(final_data)} items to {OUTPUT_FILENAME}...")
            os.makedirs(OUTPUT_DIR, exist_ok=True)
            with open(OUTPUT_FILENAME, 'w', newline='', encoding='utf-8') as f:
                writer = csv.DictWriter(f, fieldnames=['major', 'category', 'sub_category', 'item'])
                writer.writeheader()
                writer.writerows(final_data)
            print("Scraping completed successfully!")