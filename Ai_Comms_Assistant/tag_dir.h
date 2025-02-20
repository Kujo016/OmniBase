#define _CRT_SECURE_NO_WARNINGS 

#ifndef DIR_TAG_H
#define DIR_TAG_H

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
#include <future>
#include <unordered_set>
#include "kernel.cuh"
#include <cstdlib>

namespace fs = std::filesystem;


// Function declarations
void get_list_directory_full();
std::unordered_map<std::string, std::vector<std::string>> load_tags(const std::string& filename);
std::vector<std::string> read_txt(const std::string& filepath);
std::string removeInvalidUtf8(const std::string& input);
void process_directory(const std::string& directory, const std::string& tag_file);
std::unordered_set<std::string> load_tags_from_code(const std::string& filename);
void process_directory_code(const std::string& directory, const std::string& outputFile);
std::string to_lower(const std::string& input);
json process_batch_return(const std::vector<std::string>& files, const std::unordered_set<std::string>& tagSet);
std::unordered_map<std::string, std::vector<std::string>> extract_tags_from_file(const std::string& file, const std::unordered_set<std::string>& tagSet);
#endif // DIR_TAG_H

