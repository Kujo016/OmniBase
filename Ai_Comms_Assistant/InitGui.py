import os
import ctypes

def run_cpp_function():
    os.add_dll_directory(r"C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v12.8\bin")
    os.add_dll_directory(r"C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v12.8\lib\x64")

    # Get the script's actual directory
    script_dir = os.path.dirname(os.path.abspath(__file__))

    # Construct the correct DLL path
    dll_path = os.path.join(script_dir, "mylib_cud.dll")

    if not os.path.exists(dll_path):
        raise FileNotFoundError(f"DLL file not found at path: {dll_path}")

    # Load the DLL
    mylib = ctypes.CDLL(dll_path)

    # Check for function
    if not hasattr(mylib, 'run'):
        raise AttributeError("Function 'run' not found in the DLL.")

    # Construct paths
    directory = os.path.join(script_dir, "AI", "targetFile").encode('utf-8')

    tag_file = b"tags\\tags.txt"
    code_file = b"tags\\code_tags.txt"

    mylib.run(directory, tag_file, code_file)

if __name__ == "__main__":
    run_cpp_function()
