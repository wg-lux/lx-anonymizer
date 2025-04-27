# Define a function which creates a file (YYYY-MM-DD_HH-MM-SS.txt) in the directory "./data" and writes a message to it
import subprocess
import platform

def write_message(message=None):
    import os
    import datetime

    # Create a directory if it does not exist
    os.makedirs("data", exist_ok=True)
    os.makedirs("logs", exist_ok=True)

    if not message:
        message = "SUCCESS"

    # Get the current date and time
    now = datetime.datetime.now()
    date_time = now.strftime("%Y-%m-%d_%H-%M-%S")

    # Create a file with the current date and time
    file_name = f"logs/{date_time}.txt"
    with open(file_name, "w") as file:
        file.write(message)

    return file_name

def main():
    print("Hello from DevEnv for lx-anonymizer project!")
    write_message()


if __name__ == "__main__":
    main()
