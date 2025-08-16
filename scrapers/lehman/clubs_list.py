# Import required libraries
import requests
import csv
from bs4 import BeautifulSoup
import os

# This is the hidden API endpoint that provides the club data in JSON format.
# The 'top=5000' parameter asks the server to send up to 5000 clubs.
API_URL = "https://lehman.campuslabs.com/engage/api/discovery/search/organizations?top=5000"
CLUB_BASE_URL = "https://lehman.campuslabs.com/engage/organization/"
OUTPUT_DIR = "data_demo/lehman/clubs"
OUTPUT_FILENAME = os.path.join(OUTPUT_DIR, "lehman_clubs_data.csv")

HEADERS = {
    'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36'
}

def scrape_lehman_clubs_api():
    print(f"Fetching data from API: {API_URL}...")
    try:
        response = requests.get(API_URL, headers=HEADERS)
        response.raise_for_status()

        # Parse the JSON data from the API response.
        data = response.json()
        
        # Extract the list of clubs from the JSON structure.
        # The actual club data is inside the 'value' key.
        clubs = data.get('value', [])

        if not clubs:
            print("API response did not contain any clubs. Check the 'value' key in the JSON.")
            return

        scraped_data = []
        # Loop through each club in the JSON list.
        for club in clubs:
            website_key = club.get('WebsiteKey')

            page_url = f"{CLUB_BASE_URL}{website_key}" if website_key else "Not Found"

            # Extract the name and description from the JSON object for each club.
            name = club.get('Name', 'No Name Found')
            
            # Get the raw description which may contain HTML
            raw_description = club.get('Description')

            # Use BeautifulSoup to parse the HTML and get only the clean text
            if raw_description:
                soup = BeautifulSoup(raw_description, 'lxml')
                description = soup.get_text(strip = True)
            else:
                description = "No Description Found"
            
            scraped_data.append({
                'club_name': name,
                'club_description': description,
                'page_url': page_url
            })

        # Save the clean data to a CSV file.
        print(f"Saving details for {len(scraped_data)} clubs to {OUTPUT_FILENAME}...")
        os.makedirs(OUTPUT_DIR, exist_ok = True)
        with open(OUTPUT_FILENAME, 'w', newline = '', encoding = 'utf-8') as f:
            writer = csv.DictWriter(f, fieldnames = ['club_name', 'club_description', 'page_url'])
            writer.writeheader()
            writer.writerows(scraped_data)
        
        print("Scraping completed successfully!")

    except requests.exceptions.RequestException as e:
        print(f"Error fetching the API URL: {e}")
    except Exception as e:
        print(f"An error occurred: {e}")

if __name__ == "__main__":
    scrape_lehman_clubs_api()