#ifndef KERNEL_CUH
#define KERNEL_CUH

#include <cuda_runtime.h>
#include <iostream>
#include <fstream>
#include <vector>
#include <unordered_map>
#include <string>
#include <sstream>
#include <nlohmann/json.hpp> 
#include <algorithm>
#include <cctype>
#include <regex>
#include <codecvt>
#include <locale>
#include <filesystem>

#define MAX_SENTENCES 1024
#define MAX_WORDS 1024
#define MAX_SUMMARY_SENTENCES 5
#define MAX_LINE_LENGTH 1024
#define MAX_KEYWORD_LENGTH 32
#define MAX_KEYWORDS 100;
#define _CRT_SECURE_NO_WARNINGS


using json = nlohmann::json;



// CUDA kernel for calculating sentence scores
__global__ void calculate_sentence_scores(char* text, int* scores, int num_sentences);

// Tokenization functions
std::vector<std::string> tokenize_sentences(const std::string& text);
std::vector<std::string> tokenizeWords(const std::string& text);

// Summarization function
std::string summarize_text(const std::string& file_path);

// CUDA kernel for checking if a keyword exists in a line
__device__ bool contains(const char* line, const char* keyword);

// CUDA kernel for tagging lines based on keywords
__global__ void tag_text_lines(char* lines, char* keywords, int* results, int num_lines, int num_keywords, int max_line_length);
__global__ void tag_code_lines(char* data, int size);
// Process files using CUDA
json process_text_files(const std::string& filepath, const std::unordered_map<std::string, std::vector<std::string>>& tags);
void process_code_files(std::vector<std::string>& files);

#endif // KERNEL_CUH