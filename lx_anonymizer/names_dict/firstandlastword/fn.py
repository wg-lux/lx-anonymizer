def extract_first_words(file_path):
    first_words = []
    remaining_lines = []
    try:
        with open(file_path, 'r') as file:
            for line in file:
                words = line.split()
                if words:  # Check if the line is not empty
                    first_word = words[0]
                    remaining_line = ' '.join(words[1:])  # Join the remaining words
                    first_words.append(first_word)
                    remaining_lines.append(remaining_line)
    except FileNotFoundError:
        print(f"The file at {file_path} was not found.")
    except Exception as e:
        print(f"An error occurred: {e}")

    return first_words, remaining_lines

def save_to_file(data, output_file_path):
    try:
        with open(output_file_path, 'w') as file:
            for item in data:
                file.write(item + '\n')
    except Exception as e:
        print(f"An error occurred while writing to the file: {e}")

# Example usage
if __name__ == "__main__":
    input_file_path = 'first_and_last_names_neutral_ascii.txt'  # Replace with your input file path
    first_words_output_file_path = 'first_names_neutral_ascii.txt'  # Replace with your first words output file path
    remaining_lines_output_file_path = 'fn_male_remaining.txt'  # Replace with your remaining lines output file path

    first_words, remaining_lines = extract_first_words(input_file_path)
    save_to_file(first_words, first_words_output_file_path)
    save_to_file(remaining_lines, remaining_lines_output_file_path)
    
    print(f"First words have been saved to {first_words_output_file_path}")
    print(f"Remaining lines have been saved to {remaining_lines_output_file_path}")

