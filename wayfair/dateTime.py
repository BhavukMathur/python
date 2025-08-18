from datetime import datetime

def get_date_dd_mm_yyyy():
    return datetime.now().strftime("%d-%m-%Y")

# Example usage:
if __name__ == "__main__":
    print("Current Date:", get_date_dd_mm_yyyy())
