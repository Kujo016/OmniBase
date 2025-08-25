# FileTagger – CUDA-Powered Codebase Analyzer

This project is a **CUDA-accelerated file tagger**. It scans directories of source code or text files, finds tags/keywords, and writes JSON + text summaries to a `reports/` folder. It mixes **C++/CUDA** for the heavy text processing and **Python** for directory walking and orchestration.

---

## Quick Start

You’ll need **NVIDIA CUDA Toolkit** installed (tested with CUDA 12.8, but other recent versions like 11.x should work).  

1. **Extract the zip and run the setup script**  
   ```bat
   setup.bat
   ```
   This installs prerequisites and checks your environment.

2. **Open an NVCC x64 tools shell**  
   (comes with CUDA Toolkit → "x64 Native Tools Command Prompt for VS 2022" or similar).

3. **Navigate to the project directory**  
   ```bat
   cd path\to\file_tagger
   ```

4. **Build the CUDA project**  
   ```bat
   build_nvcc.bat
   ```
   This compiles the CUDA/C++ sources into `bin/mylib_cud.dll`.

5. **Prepare your input files**  
   Create a folder named `targetFile` in the project root and dump the files you want analyzed into it:
   ```
   file_tagger/
     targetFile/
       demo_dir/
        dir_1
	 file.simpleFile
	dir_n
	 file.simpleFile
   ```
Readable files include:
.txt, .cpp, .c, .h, .hpp, .cu, .cuh,
.py, .java, .js, .jsx, .ts, .tsx,
.cs, .go, .swift ,.rs ,.kt, .kts,
.html, .htm, .css, .sh, .bash, 
.bat, .cmd, .gd

6. **Run the sorter bot**  
   ```bash
   python init_sorter_bot.py
   ```
   This loads the compiled CUDA DLL and runs the tagging logic.

7. **Check the reports**  
   Results are saved to:
   ```
   reports/
     code_summary_results.json
     directory_structure_full.txt
   ```

---

## Project Structure

- **kernel.cu / kernel.cuh** – CUDA kernels for text processing.
- **tag_dir.h** – Core API for directory + text file processing.
- **dir_tag.cpp** – Implements scanning, JSON summaries, and report writing.
- **list_directory_full.py** – Walks the `targetFile` directory and dumps a full tree listing.
- **init_sorter_bot.py** – Python entrypoint. Loads the CUDA DLL and runs the tagging pipeline.
- **setup.bat** – Environment check (internet, Visual Studio Build Tools, vcpkg).
- **build_nvcc.bat** – Compiles the C++/CUDA code into `bin/mylib_cud.dll`.
- **run_sorter.bat** – Convenience script to launch the pipeline.

---

## How it Works

1. Python (`init_sorter_bot.py`) loads `mylib_cud.dll` with `ctypes`.
2. The DLL (`dir_tag.cpp` + `kernel.cu`) runs a recursive directory scan, filtering by valid extensions (e.g. `.cpp`, `.h`, `.py`, `.js`, `.cu`, `.cuh`).
3. Files are processed: tags are counted (via CUDA kernel) and per-line hits are recorded.
4. JSON and text reports are saved to `reports/`.

---

## Requirements

- Windows (tested on Win10/11)  
- NVIDIA GPU with CUDA Toolkit (12.8 recommended)  
- Visual Studio 2022 Build Tools  
- vcpkg (installed by setup script)  
- Python 3.11+  
