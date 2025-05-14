import os
import sys

def update_bin_files(venv_dir):
    """
    Update paths in the bin directory of a virtual environment.

    :param venv_dir: Path to the virtual environment directory.
    """
    bin_dir = os.path.join(venv_dir, "bin")
    if not os.path.exists(bin_dir):
        print(f"Error: The bin directory {bin_dir} does not exist.")
        return

    new_path = os.path.abspath(venv_dir)

    # Update all scripts in the bin directory
    for file_name in os.listdir(bin_dir):
        file_path = os.path.join(bin_dir, file_name)
        if os.path.isfile(file_path):
            # Skip python executables
            if file_name in ("python", "python3"):
                continue

            with open(file_path, "rb") as file:
                content = file.read()

            # Check if the file contains a shebang and detect old path
            old_path = None
            if content.startswith(b"#!"):
                first_line = content.splitlines()[0]
                if b"/bin/python" in first_line:
                    old_path = first_line.decode().strip()[2:-11]  # Extract old path

            if old_path:
                # Replace the old path with the new path
                new_content = content.replace(old_path.encode(), new_path.encode())
                with open(file_path, "wb") as file:
                    file.write(new_content)
                print(f"Updated: {file_path}")
            else:
                print(f"Skipping file (no python shebang found): {file_path}")

    print("All bin files have been updated successfully.")

# Example usage
if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("Usage: python update_bin_files.py <path_to_venv>")
    else:
        update_bin_files(sys.argv[1])
