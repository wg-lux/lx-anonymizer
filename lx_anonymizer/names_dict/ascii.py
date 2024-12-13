import string

def is_ascii(s):
    return all(c in string.printable for c in s)

def filter_non_ascii_lines(input_file, output_file):
    with open(input_file, 'r', encoding='utf-8') as infile:
        lines = infile.readlines()
    
    with open(output_file, 'w', encoding='utf-8') as outfile:
        for line in lines:
            if is_ascii(line):
                outfile.write(line)

if __name__ == "__main__":
    input_file = 'first_and_last_names_male.txt'
    output_file = 'first_and_last_names_male_ascii.txt'
    filter_non_ascii_lines(input_file, output_file)
    print(f"Filtered lines have been written to {output_file}")
