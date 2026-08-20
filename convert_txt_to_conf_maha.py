# read <input_file.txt> tahhat has R G B lines and saves into a "conf.maha" file with the header X Y R G B, X and Y are always 0 0

import sys
import argparse


def convert_txt_to_conf_maha(input_file: str, output_file: str) -> None:
    """
    Converts a text file containing RGB values into a .maha configuration file.
    The input file is expected to have lines with R G B values.
    The output .maha file will have X Y R G B, where X and Y are always 0 0.
    """
    try:
        with open(input_file, 'r') as infile:
            lines = infile.readlines()

        with open(output_file, 'w') as outfile:
            outfile.write("#Number of patterns\n")
            outfile.write(f"{len(lines)}\n\n")
            outfile.write("#X Y Red Green Blue\n")
            
            for line in lines:
                rgb_values = line.strip().split()
                if len(rgb_values) == 3:
                    r, g, b = rgb_values
                    outfile.write(f"0 0 {r} {g} {b}\n")
                else:
                    print(f"Warning: Skipping malformed line in {input_file}: {line.strip()}")
        print(f"Successfully converted '{input_file}' to '{output_file}'")

    except FileNotFoundError:
        print(f"Error: Input file '{input_file}' not found.")
        sys.exit(1)
    except Exception as e:
        print(f"An error occurred: {e}")
        sys.exit(1)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Convert a text file with RGB values to a .maha configuration file.")
    parser.add_argument("input_file", help="Path to the input .txt file (e.g., S_file.txt).")
    parser.add_argument("output_file", help="Path to the output .maha configuration file (e.g., conf.maha).")
    
    args = parser.parse_args()

    convert_txt_to_conf_maha(args.input_file, args.output_file)
