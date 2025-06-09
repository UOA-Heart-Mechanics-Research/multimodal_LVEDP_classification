import os
import sys
from datetime import datetime

class Logger:
    """
    A Logger class that redirects stdout to both the terminal and a log file.
    
    Attributes:
        log_file_path (str): Path to the log file where logs will be written.
    """

    def __init__(self, log_file_path):
        """
        Initialize the Logger instance.

        Args:
            log_file_path (str): Path to the log file.
        """
        self.terminal = sys.stdout  # Save the original stdout
        os.makedirs(os.path.dirname(log_file_path), exist_ok=True)  # Ensure the log file directory exists
        self.log_file = open(log_file_path, 'a')  # Open the log file in append mode

        # Write a header with the current date and time at the start of the log file
        current_time = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        self.log_file.write("\n\n" + "=" * 40 + "\n")
        self.log_file.write(f"Log started on: {current_time}\n")
        self.log_file.write("=" * 40 + "\n")
        self.flush()  # Ensure the header is written immediately

    def write(self, message):
        """
        Write a message to both the terminal and the log file.

        Args:
            message (str): The message to be logged.
        """
        self.terminal.write(message)  # Print the message to the terminal
        self.log_file.write(message)  # Write the message to the log file

    def flush(self):
        """
        Flush the output streams to ensure all data is written.
        """
        self.terminal.flush()  # Flush the terminal output
        self.log_file.flush()  # Flush the log file output