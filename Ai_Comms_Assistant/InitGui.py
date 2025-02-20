import ctypes
import os

def run_cpp_function():
    # Add the directory containing the CUDA DLLs to the search path
    os.add_dll_directory(r"c:\program files\nvidia gpu computing toolkit\cuda\v12.8\bin")
    os.add_dll_directory(r"c:\program files\nvidia gpu computing toolkit\cuda\v12.8\lib\x64")

    # Get the current working directory
    current_directory = os.getcwd()
    print(f"Current working directory: {current_directory}")

    # Construct the full path to the DLL file
    dll_filename = "mylib_cud.dll"
    dll_path = os.path.join(current_directory, dll_filename)

    # Check if the DLL file exists
    if not os.path.exists(dll_path):
        raise FileNotFoundError(f"DLL file not found at path: {dll_path}")

    # Load the shared library
    mylib = ctypes.CDLL(dll_path)

    # Print available functions to debug
    print(dir(mylib))

    # Ensure the function name matches exactly
    if not hasattr(mylib, 'run'):
        raise AttributeError("Function 'run' not found in the DLL.")

    # Set argument types
    mylib.run.argtypes = [ctypes.c_char_p, ctypes.c_char_p, ctypes.c_char_p]

    # Convert Python strings to C-compatible strings
    directory = b"c:\\ai"
    tag_file = b"tags.txt"
    code_file = b"code_tags.txt"

    # Call the C++ function
    mylib.run(directory, tag_file, code_file)

if __name__ == "__main__":
    run_cpp_function()