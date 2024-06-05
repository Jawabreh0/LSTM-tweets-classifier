import pandas as pd

def split_csv_file(input_file, split_row):
    # Load the CSV file into a DataFrame
    data = pd.read_csv(input_file)
    
    # Split the data into two parts
    data_first_part = data.iloc[:split_row]  # Rows up to split_row, including
    data_second_part = data.iloc[split_row:]  # Rows after split_row
    
    # Save the first part back to the original file
    data_first_part.to_csv(input_file, index=False)
    print(f"Updated '{input_file}' to contain only the first {split_row} rows.")
    
    # Save the second part to a new file
    output_file = "testing-normal.csv"
    data_second_part.to_csv(output_file, index=False)
    print(f"Extracted rows after {split_row} into '{output_file}'.")

# Define the file name and the row number where to split
file_name = "normal.csv"
split_row = 19999  # The first 20000 rows remain in the original, the rest go to the new file
split_csv_file(file_name, split_row)
