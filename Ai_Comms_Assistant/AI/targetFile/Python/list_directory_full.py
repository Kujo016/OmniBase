import os

def list_directory_contents(path="."):
    output = []
    total_size = 0  # Initialize total size counter
    
    for root, dirs, files in os.walk(path):
        level = root.replace(path, "").count(os.sep)
        indent = " " * 4 * level
        output.append(f"{indent}{os.path.basename(root)}/")
        sub_indent = " " * 4 * (level + 1)
        for file in files:
            file_path = os.path.join(root, file)
            file_size = os.path.getsize(file_path)
            total_size += file_size
            output.append(f"{sub_indent}{file} - {file_size} bytes")
    
    output.append(f"\nTotal size of directory: {total_size} bytes ({total_size / (1024 * 1024):.2f} MB)")
    return "\n".join(output)

if __name__ == "__main__":
    directory = os.getcwd()  # Gets the current working directory (absolute path)
    output_file = os.path.join(os.path.dirname(__file__), "directory_structure_full.txt")
    
    print(f"Listing contents of directory: {directory}\n")
    structure = list_directory_contents(directory)
    
    # Print to console
    print(structure)
    
    # Save to a .txt file with the directory path at the top and bottom
    with open(output_file, "w", encoding="utf-8") as file:
        file.write(f"Directory structure of: {directory}\n\n")
        file.write(structure)
        file.write(f"\n\nSystem path of main directory: {directory}")
    
    print(f"\nDirectory structure has been saved to: {output_file}")
