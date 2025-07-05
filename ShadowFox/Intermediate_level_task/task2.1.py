from selenium import webdriver
from selenium.webdriver.chrome.options import Options
from selenium.webdriver.common.by import By
import time
import os
import requests
from urllib.parse import urljoin, urlparse

# Ask the user for a website URL
url = input("Enter the website URL (e.g., https://quotes.toscrape.com/): ")

# Set up Selenium with headless Chrome
chrome_options = Options()
chrome_options.add_argument("--headless")
chrome_options.add_argument("--disable-gpu")
chrome_options.add_argument("--window-size=1920,1080")
chrome_options.add_argument("--no-sandbox")
chrome_options.add_argument("--disable-dev-shm-usage")

driver = webdriver.Chrome(options=chrome_options)
driver.get(url)
time.sleep(2)  # Wait for page to load

# Extract quotes (for quotes.toscrape.com)
quotes = driver.find_elements(By.CLASS_NAME, "text")
print(f"Quotes from {url}:")
for idx, quote in enumerate(quotes, 1):
    print(f"{idx}. {quote.text}")

# Extract and download images
images = driver.find_elements(By.TAG_NAME, "img")
img_folder = "downloaded_images"
os.makedirs(img_folder, exist_ok=True)
print("\nDownloading images:")
for img in images:
    img_url = img.get_attribute("src")
    if img_url:
        img_name = os.path.basename(urlparse(img_url).path)
        img_path = os.path.join(img_folder, img_name)
        try:
            img_data = requests.get(img_url).content
            with open(img_path, "wb") as f:
                f.write(img_data)
            print(f"Downloaded: {img_name}")
        except Exception as e:
            print(f"Failed to download {img_url}: {e}")

# Extract and download CSV files
links = driver.find_elements(By.TAG_NAME, "a")
print("\nDownloading CSV files:")
for link in links:
    href = link.get_attribute("href")
    if href and href.endswith('.csv'):
        csv_url = urljoin(url, href)
        csv_name = os.path.basename(urlparse(csv_url).path)
        try:
            csv_data = requests.get(csv_url).content
            with open(csv_name, "wb") as f:
                f.write(csv_data)
            print(f"Downloaded CSV: {csv_name}")
        except Exception as e:
            print(f"Failed to download {csv_url}: {e}")

driver.quit()