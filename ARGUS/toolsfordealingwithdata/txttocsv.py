import csv
from pathlib import Path

script_dir = Path(__file__).parent

def txt_to_csv(input_file_path, output_file_path, csv_file_path):
    """
    Reads two text files line by line and writes them into a CSV file
    with columns 'input' and 'response'.
    """
    with open(input_file_path, 'r', encoding='utf-8') as input_file, \
         open(output_file_path, 'r', encoding='utf-8') as output_file, \
         open(csv_file_path, 'w', newline='', encoding='utf-8') as csv_file:
        
        writer = csv.writer(csv_file)
        writer.writerow(["input", "response"])  # Write the header row

        for input_line, response_line in zip(input_file, output_file):
            input_line = input_line.strip()
            response_line = response_line.strip()
            # Only write rows that are not empty, if desired
            if input_line and response_line:
                writer.writerow([input_line, response_line])

if __name__ == "__main__":
    # Adjust file paths as needed
    input_file_path = script_dir / "data/inputtexts.txt"
    output_file_path = script_dir / "data/outputtexts.txt"
    csv_file_path = script_dir / "conversation_pairs.csv"
    
    txt_to_csv(input_file_path, output_file_path, csv_file_path)
    print(f"CSV file '{csv_file_path}' created successfully.")