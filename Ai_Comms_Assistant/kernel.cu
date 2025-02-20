#include "kernel.cuh"
#include "tag_dir.h"



__global__ void calculate_sentence_scores(char* text, int* scores, int num_sentences) {
    int idx = threadIdx.x;
    if (idx < num_sentences) {
        int score = 0;
        for (int i = 0; i < MAX_WORDS; i++) {
            if (text[idx * MAX_WORDS + i] == '\0') break;
            if (text[idx * MAX_WORDS + i] == ' ') score++;
        }
        scores[idx] = score;
    }
}

std::vector<std::string> tokenize_sentences(const std::string& text) {
    // Remove punctuation by replacing non-word and non-space characters with a space.
    std::string cleaned = std::regex_replace(text, std::regex(R"([^\w\s])"), " ");
    std::istringstream iss(cleaned);
    std::vector<std::string> tokens;
    std::string word;
    while (iss >> word) {
        // Optionally convert to lowercase for case-insensitive matching.
        std::transform(word.begin(), word.end(), word.begin(), ::tolower);
        tokens.push_back(word);
    }
    return tokens;
}
std::vector<std::string> tokenizeWords(const std::string& text) {
    // Replace any character that isn't a word character or whitespace with a space.
    std::string cleaned = std::regex_replace(text, std::regex(R"([^\w\s])"), " ");
    std::istringstream iss(cleaned);
    std::vector<std::string> tokens;
    std::string word;
    while (iss >> word) {
        std::transform(word.begin(), word.end(), word.begin(), ::tolower);
        tokens.push_back(word);
    }
    return tokens;
}
std::string summarize_text(const std::string& file_path) {
    std::ifstream file(file_path);
    if (!file) return "Error opening file.";

    std::string text((std::istreambuf_iterator<char>(file)), std::istreambuf_iterator<char>());
    file.close();

    std::vector<std::string> sentences = tokenize_sentences(text);
    int num_sentences = sentences.size();

    if (num_sentences == 0) return "File is empty.";

    char* d_text;
    int* d_scores;
    int h_scores[MAX_SENTENCES] = { 0 };

    cudaMalloc((void**)&d_text, MAX_SENTENCES * MAX_WORDS * sizeof(char));
    cudaMalloc((void**)&d_scores, MAX_SENTENCES * sizeof(int));

    cudaMemcpy(d_text, text.c_str(), text.size() * sizeof(char), cudaMemcpyHostToDevice);

    calculate_sentence_scores << <1, num_sentences >> > (d_text, d_scores, num_sentences);
    cudaMemcpy(h_scores, d_scores, num_sentences * sizeof(int), cudaMemcpyDeviceToHost);

    cudaFree(d_text);
    cudaFree(d_scores);

    std::vector<std::pair<int, std::string>> sorted_sentences;
    for (int i = 0; i < num_sentences; i++) {
        sorted_sentences.emplace_back(h_scores[i], sentences[i]);
    }

    std::sort(sorted_sentences.begin(), sorted_sentences.end(), std::greater<>());

    std::string summary;
    for (int i = 0; i < std::min((int)sorted_sentences.size(), MAX_SUMMARY_SENTENCES); i++) {
        summary += sorted_sentences[i].second + ". ";
    }

    return summary;
}


// GPU Kernel: Checks if a keyword exists in each line
__device__ bool contains(const char* line, const char* keyword) {
    int i = 0;
    while (line[i] != '\0') {
        int j = 0;
        while (line[i + j] != '\0' && keyword[j] != '\0' && line[i + j] == keyword[j]) {
            j++;
        }
        if (keyword[j] == '\0') {
            return true;  // Found keyword
        }
        i++;
    }
    return false;
}

__global__ void tag_text_lines(char* lines, char* keywords, int* results, int num_lines, int num_keywords, int max_line_length) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;

    if (idx < num_lines) {
        char* line = lines + idx * max_line_length;

        for (int k = 0; k < num_keywords; k++) {
            char* keyword = keywords + k * 32; // Assuming max keyword length of 32
            if (contains(line, keyword)) { // Use custom function instead of strstr()
                results[idx] = k + 1; // Store keyword index if found
            }
        }
    }
}

// Process files using CUDA
__host__ json process_text_files(const std::string& filepath, const std::unordered_map<std::string, std::vector<std::string>>& tags) {
    std::cout << "Processing File: " << filepath << std::endl;
    std::vector<std::string> lines = read_txt(filepath);

    int num_lines = static_cast<int>(lines.size());
    if (num_lines == 0 || tags.empty()) return json();

    // Flatten the tag list into keyword and category arrays
    std::vector<std::string> keyword_list;
    std::vector<std::string> category_list;
    for (const auto& kv : tags) {
        const std::string& category = kv.first;
        for (const std::string& keyword : kv.second) {
            std::string lower_keyword = keyword;
            std::transform(lower_keyword.begin(), lower_keyword.end(), lower_keyword.begin(), ::tolower);
            keyword_list.push_back(lower_keyword);
            category_list.push_back(category);
        }
    }
    int num_keywords = static_cast<int>(keyword_list.size());
    if (num_keywords == 0) {
        std::cerr << "Error: No keywords were loaded from categories!" << std::endl;
        return json();
    }

    // Declare all variables at the start to avoid skipping initialization
    char* h_lines = nullptr;
    char* h_keywords = nullptr;
    int* h_results = nullptr;
    char* d_lines = nullptr;
    char* d_keywords = nullptr;
    int* d_results = nullptr;
    json output;
    json summary;
    bool foundMatch = false;

    // Allocate CPU memory
    h_lines = new char[num_lines * MAX_LINE_LENGTH];
    h_keywords = new char[num_keywords * 32]; // Assuming max keyword length of 32
    h_results = new int[num_lines];

    // Zero out memory BEFORE copying data
    memset(h_lines, 0, num_lines * MAX_LINE_LENGTH);
    memset(h_results, 0, num_lines * sizeof(int));
    memset(h_keywords, 0, num_keywords * 32);

    // Copy lines into fixed-size buffers (convert to lowercase)
    for (int i = 0; i < num_lines; i++) {
        std::string lower_line = lines[i];
        std::transform(lower_line.begin(), lower_line.end(), lower_line.begin(), ::tolower);
        strncpy(h_lines + i * MAX_LINE_LENGTH, lower_line.c_str(), MAX_LINE_LENGTH - 1);
    }

    // Copy keywords into fixed-size buffers (Ensure null termination)
    for (int i = 0; i < num_keywords; i++) {
        strncpy(h_keywords + i * 32, keyword_list[i].c_str(), 31);
        h_keywords[i * 32 + 31] = '\0'; // Force null termination
    }

    // Allocate GPU memory
    cudaError_t err;

    std::cout << "DEBUG: Attempting to allocate memory for d_lines" << std::endl;
    err = cudaMalloc(&d_lines, num_lines * MAX_LINE_LENGTH);
    if (err != cudaSuccess) {
        std::cerr << "CUDA malloc failed for d_lines: " << cudaGetErrorString(err) << std::endl;
        std::cerr << "DEBUG: Exiting at STEP 1 - Memory Allocation" << std::endl;
        return 3;
    }

    std::cout << "DEBUG: Attempting to allocate memory for d_keywords" << std::endl;
    err = cudaMalloc(&d_keywords, num_keywords * 32);
    if (err != cudaSuccess) {
        std::cerr << "CUDA malloc failed for d_keywords: " << cudaGetErrorString(err) << std::endl;
        std::cerr << "DEBUG: Exiting at STEP 2 - Memory Allocation" << std::endl;
        return 3;
    }

    std::cout << "DEBUG: Attempting to allocate memory for d_results" << std::endl;
    err = cudaMalloc(&d_results, num_lines * sizeof(int));
    if (err != cudaSuccess) {
        std::cerr << "CUDA malloc failed for d_results: " << cudaGetErrorString(err) << std::endl;
        std::cerr << "DEBUG: Exiting at STEP 3 - Memory Allocation" << std::endl;
        return 3;
    }

    std::cout << "DEBUG: Successfully allocated CUDA memory." << std::endl;


    // Define these variables **before kernel launch** to fix "bypassing initialization" error
    int threadsPerBlock = 256;
    int maxGridSize;
    cudaDeviceGetAttribute(&maxGridSize, cudaDevAttrMaxGridDimX, 0);
    int blocksPerGrid = (num_lines + threadsPerBlock - 1) / threadsPerBlock;
    blocksPerGrid = std::min(blocksPerGrid, maxGridSize);

    // Copy data to GPU
    cudaMemcpy(d_lines, h_lines, num_lines * MAX_LINE_LENGTH, cudaMemcpyHostToDevice);
    cudaMemcpy(d_keywords, h_keywords, num_keywords * 32, cudaMemcpyHostToDevice);
    cudaMemcpy(d_results, h_results, num_lines * sizeof(int), cudaMemcpyHostToDevice);

    
    
    // Validate CUDA kernel launch parameters
    if (blocksPerGrid <= 0 || threadsPerBlock <= 0) {
        std::cerr << "Invalid kernel launch configuration: blocksPerGrid = " << blocksPerGrid << std::endl;
        return 3;
    }

    std::cout << "DEBUG: Launching CUDA Kernel with " << blocksPerGrid << " blocks and " << threadsPerBlock << " threads per block." << std::endl;
    
    // Print memory info before running CUDA
    size_t free_mem, total_mem;
    cudaMemGetInfo(&free_mem, &total_mem);
    std::cout << "CUDA Memory Status: Free = " << free_mem << " / Total = " << total_mem << std::endl;

    //Launch
    tag_text_lines << <blocksPerGrid, threadsPerBlock >> > (d_lines, d_keywords, d_results, num_lines, num_keywords, MAX_LINE_LENGTH);

    // Check for launch errors before synchronization
    cudaError_t launchError = cudaGetLastError();
    if (err != cudaSuccess) {
        std::cerr << "CUDA Kernel Error: " << cudaGetErrorString(err) << std::endl;
        return json(); // Ensure the error is logged properly
    }
    if (launchError != cudaSuccess) {
        std::cerr << "CUDA Kernel Launch Error: " << cudaGetErrorString(launchError) << std::endl;
        std::cerr << "DEBUG: Exiting at STEP 4 - Kernel Launch" << std::endl;

        return json();
    }

    std::cout << "DEBUG: CUDA Kernel Launched Successfully. Now synchronizing..." << std::endl;

    // Synchronize and check for execution errors
    cudaError_t kernel_err = cudaDeviceSynchronize();
    if (kernel_err != cudaSuccess) {
        std::cerr << "CUDA Kernel Execution Failed: " << cudaGetErrorString(kernel_err) << std::endl;
        std::cerr << "DEBUG: Exiting at STEP 5 - Kernel Execution" << std::endl;
        return 3;
    }

    std::cout << "DEBUG: CUDA Kernel Executed Successfully." << std::endl;


    // Copy results back from GPU
    cudaMemcpy(h_results, d_results, num_lines * sizeof(int), cudaMemcpyDeviceToHost);

    std::cout << "DEBUG: CUDA Results copied back. Checking first 10 values..." << std::endl;
    std::cout << "Checking CUDA Results: h_results first 10 values = ";
    for (int i = 0; i < std::min(num_lines, 10); i++) {
        if (h_results[i] < 0 || h_results[i] >= num_keywords) {
            std::cerr << "Warning: h_results[" << i << "] has invalid index " << h_results[i] << std::endl;
            continue;
        }
    }
    std::cout << std::endl;




    // Populate JSON with matches before freeing memory
    for (int i = 0; i < num_lines; i++) {
        // Remove any invalid UTF-8 sequences from the line
        std::string sanitizedLine = removeInvalidUtf8(lines[i]);
        if (h_results[i] >= 0 && h_results[i] < num_keywords) {
            foundMatch = true;
            std::string matched_category = category_list[h_results[i]];
            summary[matched_category].push_back(sanitizedLine);
        }
        else if (h_results[i] > num_keywords) {
            std::cerr << "Warning: Invalid index " << h_results[i] << " for category_list." << std::endl;
        }
    }

    // Prevent empty JSON if no matches found
    if (!foundMatch) {
        std::cerr << "No matches found in this file!\n";
        output["error"] = "No matches found";
    }
    else {
        try {
            std::cout << "DEBUG: Constructing JSON output..." << std::endl;

            output["summary"] = summary;

            std::cout << "DEBUG: JSON Output Successfully Constructed." << std::endl;
        }
        catch (const std::exception& e) {
            std::cerr << "JSON Processing Error: " << e.what() << std::endl;
            return 3;
        }

    }

    // **Only free GPU memory after processing is done**
    cudaFree(d_lines);
    cudaFree(d_keywords);
    cudaFree(d_results);

    // **Free CPU memory after JSON is built**
    delete[] h_lines;
    delete[] h_keywords;
    delete[] h_results;

    return output;
}
/////////////////////////////////////////////////////////////////////////////////////////////////////////////////
// CUDA Kernel 
__global__ void tag_code_lines(char* data, int size) {
    int idx = threadIdx.x + blockIdx.x * blockDim.x;

    if (idx < size - 6) //Buffering the index
    {
        if (data[idx] == '#' && (data[idx + 1] == ' ' || data[idx + 1] == '\t')) {

        }
    }
}

// Process files with CUDA
__host__ void process_code_files(std::vector<std::string>& files) {
    std::cout << "Processing Python Files With Cuda " << std::endl;
    for (auto& file : files) {
        std::ifstream inFile(file, std::ios::binary | std::ios::ate);
        if (!inFile) {
            std::cerr << "Error opening file: " << file << std::endl;
            continue;
        }

        size_t fileSize = inFile.tellg();
        inFile.seekg(0, std::ios::beg);

        if (fileSize == 0) {
            std::cerr << "Skipping empty file: " << file << std::endl;
            continue;
        }

        std::vector<char> fileData(fileSize);
        inFile.read(fileData.data(), fileSize);
        inFile.close();

        // Allocate CUDA memory
        char* d_data;
        if (cudaMalloc((void**)&d_data, fileSize) != cudaSuccess) {
            std::cerr << "CUDA malloc failed for file: " << file << std::endl;
            continue;
        }
        std::cout << "cudaMalloc: Stage 1 completed" << std::endl;

        if (cudaMemcpy(d_data, fileData.data(), fileSize, cudaMemcpyHostToDevice) != cudaSuccess) {
            std::cerr << "CUDA memcpy (HostToDevice) failed for file: " << file << std::endl;
            cudaFree(d_data);
            continue;
        }

        // Launch CUDA kernel
        int blockSize = 256;
        int gridSize = (fileSize + blockSize - 1) / blockSize;

        // Ensure we don’t exceed CUDA’s max grid size
        int maxGridSize;
        cudaDeviceGetAttribute(&maxGridSize, cudaDevAttrMaxGridDimX, 0);
        gridSize = std::min(gridSize, maxGridSize);

        tag_code_lines << <gridSize, blockSize >> > (d_data, fileSize);
        cudaError_t err = cudaDeviceSynchronize();
        if (err != cudaSuccess) {
            std::cerr << "CUDA Kernel failed for file: " << file
                << " with error: " << cudaGetErrorString(err) << std::endl;
            cudaFree(d_data);
            continue;
        }

        // Copy back processed data
        if (cudaMemcpy(fileData.data(), d_data, fileSize, cudaMemcpyDeviceToHost) != cudaSuccess) {
            std::cerr << "CUDA memcpy (DeviceToHost) failed for file: " << file << std::endl;
            cudaFree(d_data);
            continue;
        }

        cudaFree(d_data); // Always free CUDA memory

        // Write processed file
        std::ofstream outFile(file, std::ios::binary);
        if (!outFile) {
            std::cerr << "Error writing file: " << file << std::endl;
            continue;
        }
        outFile.write(fileData.data(), fileSize);
        outFile.close();
    }
}
