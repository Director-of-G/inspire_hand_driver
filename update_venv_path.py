import os
import sys

def update_venv_path(venv_dir):
    """
    Update paths in a Python virtual environment to reflect the new location.
    
    :param venv_dir: Path to the virtual environment directory.
    """
    venv_dir = os.path.abspath(venv_dir)  # Ensure the path is absolute

    if not os.path.exists(venv_dir):
        print(f"Error: The directory {venv_dir} does not exist.")
        return

    # Detect and update activate scripts
    activate_scripts = [
        os.path.join(venv_dir, "bin", "activate"),
        os.path.join(venv_dir, "bin", "activate.csh"),
        os.path.join(venv_dir, "bin", "activate.fish"),
    ]
    current_python_path = os.path.join(venv_dir, "bin", "python")

    for script in activate_scripts:
        if os.path.exists(script):
            with open(script, "r") as file:
                content = file.read()

            # Find the line that sets VIRTUAL_ENV and update its path to absolute path
            if script.endswith("activate"):
                if 'VIRTUAL_ENV=' in content:
                    start_index = content.find('VIRTUAL_ENV="') + len('VIRTUAL_ENV="')
                    end_index = content.find('"', start_index)
                    old_path = content[start_index:end_index]
                    
                    # Update with the absolute path
                    content = content.replace(old_path, venv_dir)
            # Special case for activate.csh (setenv VIRTUAL_ENV)
            if script.endswith("activate.csh"):
                # Ensure we are updating the correct setenv line
                if 'setenv VIRTUAL_ENV' in content:
                    # Only modify the setenv VIRTUAL_ENV line
                    lines = content.splitlines()
                    for i, line in enumerate(lines):
                        if line.startswith('setenv VIRTUAL_ENV'):
                            parts = line.split('"')[-2]  # Get the path part
                            old_path = parts[1]
                            lines[i] = f'setenv VIRTUAL_ENV "{venv_dir}"'  # Update with the new path
                            print(f"Updated setenv VIRTUAL_ENV in: {script}")
                            break
                    content = "\n".join(lines)


            # Special case for activate.fish (set -gx VIRTUAL_ENV)    
            if script.endswith("activate.fish"):
                if 'set -gx VIRTUAL_ENV' in content:
                    # Only modify the set VIRTUAL_ENV line
                    lines = content.splitlines()
                    for i, line in enumerate(lines):
                        if line.startswith('set -gx VIRTUAL_ENV'):
                            parts = line.split('"')[-2]  # Get the path part
                            lines[i] = f'set -gx VIRTUAL_ENV "{venv_dir}"'  # Update with the new path
                            print(f"Updated set -gx VIRTUAL_ENV in: {script}")
                            break
                    content = "\n".join(lines)
            with open(script, "w") as file:
                file.write(content)
            print(f"Updated VIRTUAL_ENV path in: {script}")

    # Update pyvenv.cfg
    pyvenv_cfg_path = os.path.join(venv_dir, "pyvenv.cfg")
    if os.path.exists(pyvenv_cfg_path):
        with open(pyvenv_cfg_path, "r") as file:
            lines = file.readlines()
        with open(pyvenv_cfg_path, "w") as file:
            for line in lines:
                if line.startswith("home = "):
                    python_home = os.path.dirname(sys.executable)
                    file.write(f"home = {python_home}\n")
                elif line.startswith("include-system-site-packages = "):
                    file.write(line)
                else:
                    file.write(line.replace(venv_dir, os.path.abspath(venv_dir)))
        print(f"Updated: {pyvenv_cfg_path}")

    print("Virtual environment paths have been updated successfully.")

# Example usage
if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("Usage: python update_venv_path.py <path_to_venv>")
    else:
        update_venv_path(sys.argv[1])

