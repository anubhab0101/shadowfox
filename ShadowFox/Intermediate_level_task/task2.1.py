import requests
from bs4 import BeautifulSoup

def scrape_news(news_url):
    """
    Fetches and displays the latest news headlines from a given news URL.
    This function now supports multiple news sites by checking the URL.

    Args:
        news_url (str): The URL to the news page.
    """
    cleaned_url = news_url.strip().rstrip('?')
    print(f"\nAttempting to fetch news from {cleaned_url}...")

    try:
        headers = {
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/126.0.0.0 Safari/537.36'
        }
        response = requests.get(cleaned_url, headers=headers, timeout=15)
        response.raise_for_status()
        print("Successfully fetched the news page.")

        soup = BeautifulSoup(response.content, 'html.parser')
        
        print(f"\n--- Latest News Headlines from {cleaned_url} ---")
        
        headlines_found = False
        
        if 'hindustantimes.com' in cleaned_url:
            news_headlines = soup.find_all('h3', class_='hdg3')
            if news_headlines:
                headlines_found = True
                for i, story in enumerate(news_headlines, 1):
                    if story.a:
                        headline = story.a.get_text(strip=True)
                        link = story.a['href']
                        if not link.startswith('http'):
                            link = f"https://www.hindustantimes.com{link}"
                        print(f"\n{i}. {headline}")
                        print(f"   Link: {link}")

        elif 'aajtak.in' in cleaned_url:
            all_links = soup.find_all('a', title=True)
            if all_links:
                headlines_found = True
                count = 1
                printed_headlines = set()
                for link_tag in all_links:
                    headline = link_tag['title'].strip()
                    link = link_tag['href']
                    if headline and link.startswith('https://www.aajtak.in/') and 'videos' not in link:
                        if headline not in printed_headlines:
                            print(f"\n{count}. {headline}")
                            print(f"   Link: {link}")
                            printed_headlines.add(headline)
                            count += 1
        
        elif 'abplive.com' in cleaned_url:
            all_links = soup.find_all('a', title=True)
            if all_links:
                headlines_found = True
                count = 1
                printed_headlines = set()
                for link_tag in all_links:
                    headline = link_tag['title'].strip()
                    link = link_tag['href']
                    if headline and link.startswith('https://'):
                        if headline not in printed_headlines:
                            print(f"\n{count}. {headline}")
                            print(f"   Link: {link}")
                            printed_headlines.add(headline)
                            count += 1

        if not headlines_found:
            print("\nNo news articles found. The website structure may have changed or is not compatible.")
            print("This scraper is optimized for 'hindustantimes.com', 'aajtak.in', and 'abplive.com'.")
            return
            
        print("\n" + "-" * 40)

    except requests.exceptions.RequestException as e:
        print(f"An error occurred during the HTTP request for news: {e}")
    except Exception as e:
        print(f"An unexpected error occurred while scraping news: {e}")


if __name__ == "__main__":
    default_news_url = "https://www.hindustantimes.com/"

    while True:
        print("\nWeb Scraper for News Headlines")
        print("-" * 30)
        print("Choose an option:")
        print("1: Use default news source (Hindustan Times)")
        print("2: Enter a news URL manually")
        print("3: Exit")

        choice = input("Enter your choice (1, 2, or 3): ").strip()

        if choice == '1':
            print(f"\nUsing default URL: {default_news_url}")
            scrape_news(default_news_url)

        elif choice == '2':
            news_url_input = input("\nEnter the news URL to scrape: ").strip()
            if news_url_input:
                scrape_news(news_url_input)
            else:
                print("\nNo URL entered.")
        
        elif choice == '3':
            print("\nExiting the program. Goodbye!")
            break

        else:
            print("\nInvalid choice. Please enter 1, 2, or 3.")
