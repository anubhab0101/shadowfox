import os
import sys
import tkinter as tk
from tkinter import ttk, messagebox, simpledialog, filedialog
from selenium import webdriver
from selenium.webdriver.chrome.options import Options
from selenium.webdriver.common.by import By
import time
import requests
from urllib.parse import urljoin, urlparse
import webbrowser
import base64
import threading

def run_scraper():
    url = url_entry.get().strip()
    data_choice = data_type.get()
    output_text.delete(1.0, tk.END)

    if not url:
        messagebox.showerror("Error", "Please enter a website URL.")
        return

    chrome_options = Options()
    chrome_options.add_argument("--headless")
    chrome_options.add_argument("--disable-gpu")
    chrome_options.add_argument("--window-size=1920,1080")
    chrome_options.add_argument("--no-sandbox")
    chrome_options.add_argument("--disable-dev-shm-usage")
    chrome_options.add_argument("--log-level=3")
    chrome_options.add_experimental_option('excludeSwitches', ['enable-logging'])

    try:
        driver = webdriver.Chrome(options=chrome_options)
        driver.get(url)
        time.sleep(2)

        if data_choice == "Quotes":
            quotes = driver.find_elements(By.CLASS_NAME, "text")
            output_text.insert(tk.END, f"Quotes from {url}:\n")
            for idx, quote in enumerate(quotes, 1):
                output_text.insert(tk.END, f"{idx}. {quote.text}\n")

        elif data_choice == "Images":
            images = driver.find_elements(By.TAG_NAME, "img")
            img_folder = "downloaded_images"
            os.makedirs(img_folder, exist_ok=True)

            img_urls = []
            for img in images:
                img_url = img.get_attribute("src")
                if img_url:
                    img_urls.append(img_url)

            if not img_urls:
                output_text.insert(tk.END, "No images found.\n")
                driver.quit()
                return

            output_text.insert(tk.END, "Found images:\n")
            for idx, img_url in enumerate(img_urls, 1):
                output_text.insert(tk.END, f"{idx}. {img_url}\n")

            def download_images(selected_indices):
                to_download = [img_urls[i] for i in selected_indices]
                for img_url in to_download:
                    if img_url.startswith("data:image/"):
                        try:
                            header, encoded = img_url.split(",", 1)
                            file_ext = header.split(";")[0].split("/")[1]
                            img_name = f"image_{to_download.index(img_url)+1}.{file_ext}"
                            img_path = os.path.join(img_folder, img_name)
                            with open(img_path, "wb") as f:
                                f.write(base64.b64decode(encoded))
                            output_text.insert(tk.END, f"Downloaded (base64): {img_name}\n")
                        except Exception as e:
                            output_text.insert(tk.END, f"Failed to decode base64 image: {e}\nImage URL: {img_url}\n")
                    else:
                        img_name = os.path.basename(urlparse(img_url).path)
                        img_path = os.path.join(img_folder, img_name)
                        try:
                            img_data = requests.get(img_url, timeout=10)
                            img_data.raise_for_status()
                            with open(img_path, "wb") as f:
                                f.write(img_data.content)
                            output_text.insert(tk.END, f"Downloaded: {img_name}\n")
                        except Exception as e:
                            output_text.insert(tk.END, f"Failed to download {img_url}: {e}\nImage URL: {img_url}\n")

            # Ask user for download option
            option = simpledialog.askstring("Image Download", "Options:\n1. Download all images\n2. Preview and download a single image by number\n3. Download multiple images by numbers (comma separated)\nEnter your choice (1/2/3):")
            if option == "1":
                download_images(list(range(len(img_urls))))
            elif option == "2":
                num = simpledialog.askinteger("Preview Image", f"Enter the image number to preview and download (1-{len(img_urls)}):")
                if num and 1 <= num <= len(img_urls):
                    img_url = img_urls[num-1]
                    webbrowser.open(img_url)
                    confirm = messagebox.askyesno("Download Image", "Do you want to download this image?")
                    if confirm:
                        download_images([num-1])
                    else:
                        output_text.insert(tk.END, "Image not downloaded.\n")
                else:
                    output_text.insert(tk.END, "Invalid number.\n")
            elif option == "3":
                nums = simpledialog.askstring("Download Images", "Enter image numbers separated by commas (e.g., 1,3,5):")
                if nums:
                    indices = []
                    for num in nums.split(","):
                        num = num.strip()
                        if num.isdigit() and 1 <= int(num) <= len(img_urls):
                            indices.append(int(num)-1)
                        else:
                            output_text.insert(tk.END, f"Invalid number: {num}\n")
                    if indices:
                        download_images(indices)
                else:
                    output_text.insert(tk.END, "No numbers entered.\n")
            else:
                output_text.insert(tk.END, "Invalid choice.\n")

        elif data_choice == "CSV files":
            links = driver.find_elements(By.TAG_NAME, "a")
            output_text.insert(tk.END, "Downloading CSV files:\n")
            for link in links:
                href = link.get_attribute("href")
                if href and href.endswith('.csv'):
                    csv_url = urljoin(url, href)
                    csv_name = os.path.basename(urlparse(csv_url).path)
                    try:
                        csv_data = requests.get(csv_url).content
                        with open(csv_name, "wb") as f:
                            f.write(csv_data)
                        output_text.insert(tk.END, f"Downloaded CSV: {csv_name}\n")
                    except Exception as e:
                        output_text.insert(tk.END, f"Failed to download {csv_url}: {e}\n")
        else:
            output_text.insert(tk.END, "Invalid choice.\n")

        driver.quit()
    except Exception as e:
        output_text.insert(tk.END, f"Error: {e}\n")

def start_scraper_thread():
    threading.Thread(target=run_scraper, daemon=True).start()

root = tk.Tk()
root.title("Website Data Scraper")

frame = ttk.Frame(root, padding=10)
frame.pack(fill=tk.BOTH, expand=True)

ttk.Label(frame, text="Enter the website URL:").grid(row=0, column=0, sticky="w")
url_entry = ttk.Entry(frame, width=60)
url_entry.grid(row=0, column=1, padx=5, pady=5)

ttk.Label(frame, text="Data type:").grid(row=1, column=0, sticky="w")
data_type = ttk.Combobox(frame, values=["Quotes", "Images", "CSV files"], state="readonly")
data_type.current(0)
data_type.grid(row=1, column=1, padx=5, pady=5, sticky="w")

scrape_btn = ttk.Button(frame, text="Scrape", command=start_scraper_thread)
scrape_btn.grid(row=2, column=0, columnspan=2, pady=10)

output_text = tk.Text(frame, height=20, width=80)
output_text.grid(row=3, column=0, columnspan=2, pady=5)

root.mainloop()