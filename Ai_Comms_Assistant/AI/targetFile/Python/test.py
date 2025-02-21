import os  # NOTE: Used for file operations
import json  # NOTE: Handles JSON serialization

# TODO: Implement the main function
# FIXME: Fix the bug in data processing
# BUG: There's an issue with input validation

def process_data(data):
    """Processes the input data."""
    try:
        # TODO: Optimize the loop performance
        for item in data:
            if not isinstance(item, int):  # BUG: Crashes when input is empty
                raise ValueError("Invalid input type")
        return sum(data)
    except ValueError as e:
        print(f"Error: {e}")

def main():
    """Main execution function."""
    sample_data = [1, 2, 3, "four", 5]  # BUG: Invalid type included
    result = process_data(sample_data)
    print(f"Processed Result: {result}")

if __name__ == "__main__":
    main()
