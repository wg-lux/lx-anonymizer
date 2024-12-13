def extract_last_words(file_path):
    last_words = []
    remaining_lines = []
    try:
        with open(file_path, 'r') as file:
            for line in file:
                words = line.split()
                if words:  # Check if the line is not empty
                    last_word = words[-1]
                    remaining_line = ' '.join(words[:-1])  # Join the words except the last one
                    last_words.append(last_word)
                    remaining_lines.append(remaining_line)
    except FileNotFoundError:
        print(f"The file at {file_path} was not found.")
    except Exception as e:
        print(f"An error occurred: {e}")

    return last_words, remaining_lines

def save_to_file(data, output_file_path):
    try:
        with open(output_file_path, 'w') as file:
            for item in data:
                file.write(item + '\n')
    except Exception as e:
        print(f"An error occurred while writing to the file: {e}")

# Example usage
if __name__ == "__main__":
    input_file_path = 'fn_female_remaining.txt'  # Replace with your input file path
    last_words_output_file_path = 'last_names_female.txt'  # Replace with your last words output file path
    remaining_lines_output_file_path = 'fn_ln_female_remaining.txt'  # Replace with your remaining lines output file path

    last_words, remaining_lines = extract_last_words(input_file_path)
    save_to_file(last_words, last_words_output_file_path)
    save_to_file(remaining_lines, remaining_lines_output_file_path)
    
    print(f"Last words have been saved to {last_words_output_file_path}")
    print(f"Remaining lines have been saved to {remaining_lines_output_file_path}")
