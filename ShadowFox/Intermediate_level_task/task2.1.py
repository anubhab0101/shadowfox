
# --- Imports ---
import os
import tkinter as tk
from tkinter import ttk, messagebox, simpledialog
from selenium import webdriver
from selenium.webdriver.chrome.options import Options
from selenium.webdriver.common.by import By
import time
import requests
from urllib.parse import urljoin, urlparse
import base64
import threading
from PIL import Image, ImageTk
import io

# --- Image Preview Function ---
def preview_image(img_url):
    preview_win = tk.Toplevel(root)
    preview_win.title("Image Preview")
    try:
        if img_url.startswith("data:image/"):
            header, encoded = img_url.split(",", 1)
            img_data = base64.b64decode(encoded)
        else:
            img_data = requests.get(img_url, timeout=10).content
        image = Image.open(io.BytesIO(img_data))
        image.thumbnail((400, 400))
        photo = ImageTk.PhotoImage(image)
        label = tk.Label(preview_win, image=photo)
        label.image = photo
        label.pack()
    except Exception as e:
        tk.Label(preview_win, text=f"Failed to preview image: {e}").pack()

# --- Main Scraper Function ---
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

            def download_image_by_index(idx):
                img_url = img_urls[idx]
                if img_url.startswith("data:image/"):
                    try:
                        header, encoded = img_url.split(",", 1)
                        file_ext = header.split(";")[0].split("/")[1]
                        img_name = f"image_{idx+1}.{file_ext}"
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

            def show_image_selector():
                selector = tk.Toplevel(root)
                selector.title("Select Image to Preview/Download")
                selector.geometry("400x450")

                tk.Label(selector, text="Enter image number to preview:").pack()
                num_entry = tk.Entry(selector)
                num_entry.pack()

                img_label = tk.Label(selector)
                img_label.pack()

                status_label = tk.Label(selector, text="")
                status_label.pack()

                def preview_and_ask():
                    num_str = num_entry.get()
                    if not num_str.isdigit() or not (1 <= int(num_str) <= len(img_urls)):
                        messagebox.showerror("Error", "Invalid image number.")
                        return
                    idx = int(num_str) - 1
                    img_url = img_urls[idx]
                    try:
                        if img_url.startswith("data:image/"):
                            header, encoded = img_url.split(",", 1)
                            img_data = base64.b64decode(encoded)
                        else:
                            img_data = requests.get(img_url, timeout=10).content
                        image = Image.open(io.BytesIO(img_data))
                        image.thumbnail((300, 300))
                        photo = ImageTk.PhotoImage(image)
                        img_label.config(image=photo)
                        img_label.image = photo
                        # Ask to download
                        if messagebox.askyesno("Download", "Download this image?"):
                            download_image_by_index(idx)
                            status_label.config(text=f"Downloaded image {num_str}.")
                        else:
                            status_label.config(text="Not downloaded.")
                    except Exception as e:
                        messagebox.showerror("Error", f"Failed to preview image: {e}")

                preview_btn = tk.Button(selector, text="Preview", command=preview_and_ask)
                preview_btn.pack(pady=5)

                close_btn = tk.Button(selector, text="Close", command=selector.destroy)
                close_btn.pack(pady=5)

            # Ask user for download option
            option = simpledialog.askstring(
                "Image Download",
                "Options:\n1. Download all images\n2. Preview and download a single image by number\n3. Download multiple images by numbers (comma separated)\nEnter your choice (1/2/3):"
            )
            if option == "1":
                for idx in range(len(img_urls)):
                    download_image_by_index(idx)
            elif option == "2":
                show_image_selector()
            elif option == "3":
                nums = simpledialog.askstring("Download Images", "Enter image numbers separated by commas (e.g., 1,3,5):")
                if nums:
                    for num in nums.split(","):
                        num = num.strip()
                        if num.isdigit() and 1 <= int(num) <= len(img_urls):
                            download_image_by_index(int(num)-1)
                        else:
                            output_text.insert(tk.END, f"Invalid number: {num}\n")
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

# --- Threaded Start ---
def start_scraper_thread():
    threading.Thread(target=run_scraper, daemon=True).start()

# --- GUI Setup ---
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