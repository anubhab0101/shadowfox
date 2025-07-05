# Write a function that takes two arguments, 145 and 'o', anduses the `format` function to return a formatted string. 
# Print the result. Try to identify the representation used.

def format_number(num, fmt):
    result = format(num, fmt)
    print(f"Formatted {num} with '{fmt}': {result}")
    return result

format_number(145, 'o')