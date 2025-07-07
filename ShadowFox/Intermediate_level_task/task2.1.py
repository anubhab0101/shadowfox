from selenium import webdriver
from selenium.webdriver.chrome.options import Options
from selenium.webdriver.common.by import By
import time
import os
import requests
from urllib.parse import urljoin, urlparse

url = input("Enter the website URL : ")

chrome_options = Options()
chrome_options.add_argument("--headless")
chrome_options.add_argument("--disable-gpu")
chrome_options.add_argument("--window-size=1920,1080")
chrome_options.add_argument("--no-sandbox")
chrome_options.add_argument("--disable-dev-shm-usage")

driver = webdriver.Chrome(options=chrome_options)
driver.get(url)
time.sleep(2)  

quotes = driver.find_elements(By.CLASS_NAME, "text")
print(f"Quotes from {url}:")
for idx, quote in enumerate(quotes, 1):
    print(f"{idx}. {quote.text}")

images = driver.find_elements(By.TAG_NAME, "img")
img_folder = "downloaded_images"
os.makedirs(img_folder, exist_ok=True)

# Collect image URLs
img_urls = []
for img in images:
    img_url = img.get_attribute("src")
    if img_url:
        img_urls.append(img_url)

# Show image URLs to the user
print("\nFound images:")
for idx, img_url in enumerate(img_urls, 1):
    print(f"{idx}. {img_url}")

print("\nOptions:")
print("1. Download all images")
print("2. Download a single image by number")
print("3. Download multiple images by numbers (comma separated)")
choice = input("Enter your choice (1/2/3): ").strip()

to_download = []
if choice == "1":
    to_download = img_urls
elif choice == "2":
    num = input("Enter the image number to download: ").strip()
    if num.isdigit() and 1 <= int(num) <= len(img_urls):
        to_download = [img_urls[int(num)-1]]
    else:
        print("Invalid number.")
elif choice == "3":
    nums = input("Enter image numbers separated by commas (e.g., 1,3,5): ").split(",")
    for num in nums:
        num = num.strip()
        if num.isdigit() and 1 <= int(num) <= len(img_urls):
            to_download.append(img_urls[int(num)-1])
        else:
            print(f"Invalid number: {num}")
else:
    print("Invalid choice.")

print("\nDownloading images:")
for img_url in to_download:
    img_name = os.path.basename(urlparse(img_url).path)
    img_path = os.path.join(img_folder, img_name)
    try:
        img_data = requests.get(img_url, timeout=10)
        img_data.raise_for_status()
        with open(img_path, "wb") as f:
            f.write(img_data.content)
        print(f"Downloaded: {img_name}")
    except Exception as e:
        print(f"Failed to download {img_url}: {e}")
        print(f"Image URL: {img_url}")

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