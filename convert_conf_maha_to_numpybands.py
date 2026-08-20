"""
reads a --input-file with
```
#X Y BAND1 BAND2 BAND3 ...
332 23 77 127 88 ...
332 23 77 127 88 ...
...
```

AND SAVES AN TXT CALLED "conf.txt" on --output-file with only the bands values
e.g.

```
77 127 88 ...
77 127 88 ...
...


"""

import argparse
import sys


def convert_conf_maha_to_numpy_bands(input_file: str, output_file: str) -> None:
    """
    Reads a .maha configuration file and extracts only the band values (RGB)
    into a new text file.
    The input file is expected to have lines like: X Y R G B ...
    The output file will have lines with only R G B ...
    """
    try:
        with open(input_file, 'r') as infile:
            lines = infile.readlines()

        band_data: list[str] = []
        for line in lines:
            line = line.strip()
            if not line or line.startswith('#'):
                continue  # Skip empty lines and comments

            parts = line.split()
            if len(parts) >= 3:  # Expect at least X, Y, and one band
                # Extract bands starting from the 3rd element (index 2)
                bands = parts[2:]
                band_data.append(" ".join(bands))
            else:
                print(f"Warning: Skipping malformed data line in {input_file}: {line}")

        if not band_data:
            print(f"Warning: No valid band data found in '{input_file}'. Output file will be empty.")
            with open(output_file, 'w') as outfile:
                pass # Create an empty file
            return

        with open(output_file, 'w') as outfile:
            for data_line in band_data:
                outfile.write(data_line + '\n')
        
        print(f"Successfully extracted band data from '{input_file}' to '{output_file}'")

    except FileNotFoundError:
        print(f"Error: Input file '{input_file}' not found.")
        sys.exit(1)
    except Exception as e:
        print(f"An error occurred: {e}")
        sys.exit(1)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Extract band values from a .maha configuration file into a plain text file.")
    parser.add_argument("--input-file", required=True, help="Path to the input .maha configuration file (e.g., conf.maha)")
    parser.add_argument("--output-file", required=True, help="Path to the output plain text file (e.g., conf.txt)")
    args = parser.parse_args()

    convert_conf_maha_to_numpy_bands(args.input_file, args.output_file) 
    
# how to use
# python convert_conf_maha_to_numpybands.py --input-file "conf.maha" --output-file "conf.txt"
