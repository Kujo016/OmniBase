# Ai_Comms_Assistant
System Setup & Build Guide for CUDA-based DLL with Python and Visual Studio

1. System Specifications & Environment

Operating System:

Windows 10/11 (Ensure x64 version is installed)

Hardware Requirements:

GPU: (Specify GPU Model for CUDA compatibility)

CPU: (Optional but useful for debugging and optimization)

RAM: Minimum 16GB recommended for smooth compilation

Storage: At least 50GB free space for Visual Studio, CUDA Toolkit, and dependencies

Development Tools & Versions:

Programming Languages & Frameworks:

Python Version: Run the following command to check your installed Python version:

python --version

C++ Standard Used: C++17

CUDA Version (Toolkit & cuDNN):

nvcc --version

where nvcc

Microsoft Visual C++ (MSVC) Version:

cl

or check in Visual Studio under Help -> About Visual Studio

Visual Studio Compiler Tools Version:

where cl

or manually locate in:

C:\Program Files\Microsoft Visual Studio\2022\Community\VC\Tools\MSVC\

2. Required Installations

1. Install Visual Studio 2022 (Community Edition)

Download from: https://visualstudio.microsoft.com/

During installation, select the following components:

Desktop development with C++

Windows SDK

C++ CMake tools for Windows

MSVC v14.x tools

2. Install NVIDIA CUDA Toolkit & cuDNN

Download the latest CUDA Toolkit from:
https://developer.nvidia.com/cuda-downloads

Install with default settings.

Add CUDA paths to PATH:

set PATH=%PATH%;C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v12.8\bin

Install cuDNN from:
https://developer.nvidia.com/cudnn

Copy cudnn64_8.dll to C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v12.8\bin

3. Install vcpkg for Dependency Management

cd C:\
git clone https://github.com/microsoft/vcpkg.git
cd vcpkg
bootstrap-vcpkg.bat
vcpkg integrate install

4. Install Required Libraries using vcpkg

vcpkg install nlohmann-json:x64-windows

3. Building the CUDA-Based DLL

1. Open x64 Native Tools Command Prompt for VS 2022

cd <working_directory>

2. Compile the C++ Source Code to Object File

cl /std:c++17 /EHsc /MD /LD /I"C:\vcpkg\installed\x64-windows\include" /I"C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v12.8\include" /c dir_tag.cpp /Fo:dir_tag.obj

3. Compile and Link CUDA Code to DLL

nvcc -ccbin "C:\Program Files\Microsoft Visual Studio\2022\Community\VC\Tools\MSVC\14.43.34808\bin\Hostx64\x64" ^
  -shared -o mylib_cud.dll kernel.cu dir_tag.obj ^
  -I"C:\vcpkg\installed\x64-windows\include" ^
  -L"C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v12.8\lib\x64" -lcudart ^
  -Xcompiler "/MD /LD /std:c++17" --expt-relaxed-constexpr -std=c++17 -gencode arch=compute_80,code=sm_80

4. Verify the DLL Export Functions

Use DLL Export Viewer (dll exp) to ensure the compiled DLL has the run function.

The exported function should be available in:

dumpbin /exports mylib_cud.dll

4. Important Configuration Settings in Visual Studio

Ensure Project Type is set to DLL instead of EXE.

Add the required Include Paths in Project Settings:

C/C++ -> General -> Additional Include Directories:

C:\vcpkg\installed\x64-windows\include
C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v12.8\include

Set the Command Line Options for CUDA Compilation:

-std=c++17 -Xcompiler="/std:c++17"

5. Debugging & Testing the DLL

Ensure DLL dependencies are met:

Dependency Walker (depends.exe)

Test execution using Python:

import ctypes
mydll = ctypes.CDLL("./mylib_cud.dll")
mydll.run()

6. Summary

Installed Visual Studio 2022 and configured for C++ & CUDA development.

Installed CUDA Toolkit 12.8, cuDNN, and vcpkg for dependency management.

Compiled C++ source code into an object file.

Used NVCC to link CUDA code into a DLL.

Verified DLL functionality using DLL Export Viewer and Dependency Walker.

Tested DLL in Python with ctypes.