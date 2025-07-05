australia = ["Sydney", "Melbourne", "Brisbane", "Perth"]
uae = ["Dubai", "Abu Dhabi", "Sharjah", "Ajman"]
india = ["Mumbai", "Bangalore", "Chennai", "Delhi"]

def get_country(city):
    if city in [c.lower() for c in australia]:
        return "Australia"
    elif city in [c.lower() for c in uae]:
        return "UAE"
    elif city in [c.lower() for c in india]:
        return "India"
    else:
        return None

city1 = input("Enter the first city (or 'exit' to quit): ")
if city1.lower() == 'exit':
    pass
else:
    city2 = input("Enter the second city (or 'exit' to quit): ")
    if city2.lower() == 'exit':
        pass
    else:
        city1_lower = city1.lower()
        city2_lower = city2.lower()

        country1 = get_country(city1_lower)
        country2 = get_country(city2_lower)

        if country1 and country2:
            if country1 == country2:
                print(f"Both cities are in {country1}")
            else:
                print("They don't belong to the same country")
        else:
            print("One or both cities are not recognized.")
