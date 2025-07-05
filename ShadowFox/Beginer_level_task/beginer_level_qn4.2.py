australia = ["Sydney", "Melbourne", "Brisbane", "Perth"]
uae = ["Dubai", "Abu Dhabi", "Sharjah", "Ajman"]
india = ["Mumbai", "Bangalore", "Chennai", "Delhi"]
city = input("Enter a city name: ")
city_lower = city.lower()
australia_lower = [c.lower() for c in australia]
uae_lower = [c.lower() for c in uae]
india_lower = [c.lower() for c in india]
if city_lower in australia_lower:
    print(f"{city} is in Australia")
elif city_lower in uae_lower:
    print(f"{city} is in UAE")
elif city_lower in india_lower:
    print(f"{city} is in India")
else:
    print(f"Sorry, I don't know which country {city} is in.")