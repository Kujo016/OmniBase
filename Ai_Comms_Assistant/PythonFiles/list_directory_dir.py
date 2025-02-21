import os

def list_directories_with_sizes(path="."):
    output = []
    total_size = 0  # Initialize total size counter for all directories

    for root, dirs, files in os.walk(path):
        dir_size = sum(os.path.getsize(os.path.join(root, f)) for f in files)  # Calculate size of current directory
        total_size += dir_size
        level = root.replace(path, "").count(os.sep)
        indent = " " * 4 * level
        output.append(f"{indent}{os.path.basename(root)}/ - {dir_size} bytes ({dir_size / (1024 * 1024):.2f} MB)")
    
    output.append(f"\nTotal size of all directories: {total_size} bytes ({total_size / (1024 * 1024):.2f} MB)")
    return "\n".join(output)

if __name__ == "__main__":
    directory = os.getcwd()  # Gets the current working directory (absolute path)
    output_file = os.path.join(os.path.dirname(__file__), "directory_structure_dir.txt")  # New filename
    
    print(f"Listing directories in: {directory}\n")
    structure = list_directories_with_sizes(directory)
    
    # Print to console
    print(structure)
    
    # Save to a .txt file with the new filename and system path included
    with open(output_file, "w") as file:
        file.write(f"Directory structure of: {directory}\n\n")
        file.write(structure)
        file.write(f"\n\nSystem path of main directory: {directory}")
    
    print(f"\nDirectory structure has been saved to: {output_file}")

