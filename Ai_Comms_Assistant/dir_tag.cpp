#include "tag_dir.h"


// Function to remove invalid UTF-8 sequences from a string
std::string removeInvalidUtf8(const std::string& input) {
    std::string output;
    size_t i = 0;
    while (i < input.size()) {
        unsigned char c = input[i];
        size_t seqLength = 0;
        if (c <= 0x7F) {
            // ASCII byte, always valid
            seqLength = 1;
        }
        else if ((c & 0xE0) == 0xC0) {
            // 2-byte sequence
            seqLength = 2;
        }
        else if ((c & 0xF0) == 0xE0) {
            // 3-byte sequence
            seqLength = 3;
        }
        else if ((c & 0xF8) == 0xF0) {
            // 4-byte sequence
            seqLength = 4;
        }
        else {
            // Invalid leading byte, skip it
            ++i;
            continue;
        }

        // Check if there are enough bytes left for the sequence
        if (i + seqLength > input.size()) break;

        bool valid = true;
        // For multi-byte sequences, validate continuation bytes
        for (size_t j = 1; j < seqLength; ++j) {
            if ((static_cast<unsigned char>(input[i + j]) & 0xC0) != 0x80) {
                valid = false;
                break;
            }
        }

        if (valid) {
            // Append valid UTF-8 sequence to output
            output.append(input, i, seqLength);
            i += seqLength;
        }
        else {
            // Skip the invalid byte and try the next one
            ++i;
        }
    }
    return output;
}


std::string to_utf8(const std::string& input) {
    std::wstring_convert<std::codecvt_utf8<wchar_t>> converter;
    try {
        return converter.to_bytes(converter.from_bytes(input));
    }
    catch (...) {
        return "";  // If conversion fails, return empty string
    }
}


// Load tags from `tags.txt`
std::unordered_map<std::string, std::vector<std::string>> load_tags(const std::string& filename) {
    std::unordered_map<std::string, std::vector<std::string>> tags;
    std::ifstream file(filename);
    std::string line;

    while (std::getline(file, line)) {
        size_t colon_pos = line.find(":");
        if (colon_pos != std::string::npos) {
            std::string category = line.substr(0, colon_pos);
            std::istringstream keywords_stream(line.substr(colon_pos + 1));
            std::string keyword;

            while (std::getline(keywords_stream, keyword, ',')) {
                tags[category].push_back(keyword);
            }
        }
    }
    return tags;
}

// Read TXT files
std::vector<std::string> read_txt(const std::string& filepath) {
    std::vector<std::string> lines;
    std::ifstream file(filepath);

    if (!file) {
        std::cerr << "Error opening file: " << filepath << std::endl;
        abort();
    }

    std::string line;
    while (std::getline(file, line)) {
        lines.push_back(line);
    }

    if (lines.empty()) {
        std::cerr << "Warning: File is empty - " << filepath << std::endl;
    }

    return lines;
}



void process_directory(const std::string& directory, const std::string& tag_file) {
    auto tags = load_tags(tag_file);
    json results;

    try {
        if (!fs::exists(directory)) {
            std::cerr << "Error: Directory does not exist - " << directory << std::endl;
            return;
        }

        fs::recursive_directory_iterator it(directory, fs::directory_options::skip_permission_denied), end;

        while (it != end) {
            try {
                if (!fs::exists(it->path())) {
                    std::cerr << "Skipping missing file/directory: " << it->path() << std::endl;
                    it.increment(std::error_code()); // Move forward safely
                    continue;
                }

                if (it->is_regular_file()) {
                    std::string ext = it->path().extension().string();
                    if (ext == ".txt") {
                        std::vector<std::string> lines = read_txt(it->path().string());

                        if (lines.empty()) {
                            std::cerr << "Skipping empty file: " << it->path().string() << std::endl;
                        }
                        else {
                            try {
                                json result = process_text_files(it->path().string(), tags);
                                if (!result.empty()) {
                                    results.push_back(result);
                                }
                            }
                            catch (const std::exception& e) {
                                std::cerr << "Error processing file " << it->path().string() << ": " << e.what() << std::endl;
                            }

                        }
                    }
                }

                std::error_code ec;
                it.increment(ec);
                if (ec) {
                    try {
                        std::error_code ec;
                        it.increment(ec);
                        if (ec) {
                            std::cerr << "Error incrementing iterator: " << ec.message() << std::endl;
                        }
                    }
                    catch (const std::filesystem::filesystem_error& e) {
                        std::cerr << "Filesystem error encountered, skipping: " << e.what() << std::endl;
                    }
                }


            }
            catch (const std::filesystem::filesystem_error& e) {
                std::cerr << "Filesystem error encountered, skipping: " << e.what() << std::endl;
                it.increment(std::error_code()); // Continue without crashing
            }
        }
    }
    catch (const std::filesystem::filesystem_error& e) {
        std::cerr << "Critical filesystem error: " << e.what() << std::endl;
        return;
    }

    std::ofstream test("test_write.tmp");
    if (!test) {
        std::cerr << "Warning: Cannot write to directory! Check permissions." << std::endl;
    }
    else {
        test.close();
        fs::remove("test_write.tmp");  // Clean up test file
    }

    


    std::ofstream output_file("C:\\AI\\summary_results.json");
    if (!output_file) {
        std::cerr << "Failed to write summary_results.json" << std::endl;
        return;
    }
    std::cout << "Attempting to write results to JSON file..." << std::endl;
    try {
        std::string utf8_text = to_utf8(results.dump(4));
        output_file << utf8_text;
        std::cout << "Results successfully written to JSON file." << std::endl;
    }
    catch (const std::exception& e) {
        std::cerr << "JSON Serialization Error: " << e.what() << std::endl;
        return;
    }

    output_file << results.dump(4);
    output_file.close();
    std::cout << "Results saved to summary_results.json" << std::endl;

    
}


// Helper function to convert a string to lowercase.
std::string to_lower(const std::string& input) {
    std::string result = input;
    std::transform(result.begin(), result.end(), result.begin(),
        [](unsigned char c) { return std::tolower(c); });
    return result;
}
std::unordered_set<std::string> load_tags_from_code(const std::string& filename) {
    std::unordered_set<std::string> tags;
    std::ifstream file(filename);
    std::string line;

    if (!file) {
        std::cerr << "[ERROR] Could not open tag file: " << filename << std::endl;
        return tags;
    }

    std::cout << "[INFO] Loading tags from: " << filename << std::endl;

    while (std::getline(file, line)) {
        if (line.empty()) continue;
        tags.insert(line);
        std::cout << "[DEBUG] Loaded tag: " << line << std::endl;
    }

    if (tags.empty()) {
        std::cerr << "[WARNING] No tags loaded from: " << filename << std::endl;
    }
    else {
        std::cout << "[INFO] Loaded " << tags.size() << " tags.\n";
    }

    return tags;
}


// Function to scan a file for matching keywords from py_tags.txt
std::unordered_map<std::string, std::vector<std::string>> extract_tags_from_file(
    const std::string& file, const std::unordered_set<std::string>& tagSet) {

    std::unordered_map<std::string, std::vector<std::string>> tag_comments;
    std::ifstream infile(file);
    std::string line;

    if (!infile) {
        std::cerr << "[ERROR] Could not open file: " << file << std::endl;
        return tag_comments;
    }

    std::cout << "[INFO] Scanning file: " << file << std::endl;

    try {
        while (std::getline(infile, line)) {
            std::cout << "[DEBUG] Reading line: " << line << std::endl;

            for (const auto& tag : tagSet) {
                size_t pos = line.find(tag);
                if (pos != std::string::npos) {
                    // Ensure the tag is not the last character in the line
                    if (pos + tag.size() >= line.size()) {
                        std::cerr << "[WARNING] Tag '" << tag << "' found at end of line, skipping comment extraction: " << line << std::endl;
                        continue;
                    }

                    std::string comment = line.substr(pos + tag.size());

                    // Ensure `comment` is not empty before calling `find_first_not_of()`
                    size_t firstValidChar = comment.find_first_not_of(" \t*:/");
                    if (firstValidChar != std::string::npos) {
                        comment = comment.substr(firstValidChar);
                        tag_comments[tag].push_back(comment);
                        std::cout << "[DEBUG] Found tag '" << tag << "' in " << file << " -> " << comment << std::endl;
                    }
                    else {
                        std::cerr << "[WARNING] Empty or invalid comment after tag '" << tag << "' in " << file << " -> Skipping line: " << line << std::endl;
                    }
                }
            }
        }
    }
    catch (const std::exception& e) {
        std::cerr << "[ERROR] Exception while reading file: " << file << " -> " << e.what() << std::endl;
        std::exit(3);
    }

    return tag_comments;
}





json process_batch_return(const std::vector<std::string>& files,
    const std::unordered_set<std::string>& tagSet) {
    json localJson;
    for (const auto& file : files) {
        auto detected_tags = extract_tags_from_file(file, tagSet);
        std::cout << "[DEBUG] Processing file: " << file
            << " - Found " << detected_tags.size() << " tags." << std::endl;
        if (!detected_tags.empty()) {
            localJson[file] = {
                {"path", file},
                {"tags", detected_tags}
            };
        }
        else {
            std::cerr << "[WARNING] No tags found in " << file << std::endl;
        }
    }
    return localJson;
}






void process_directory_code(const std::string& directory, const std::string& pytag_file) {
    json jsonData;
    std::vector<std::string> codeFiles;
    size_t batchSize = 200;  // Adjust as needed

    if (!fs::exists(directory) || !fs::is_directory(directory)) {
        std::cerr << "[ERROR] Invalid directory: " << directory << std::endl;
        return;
    }

    std::cout << "[INFO] Scanning directory: " << directory << std::endl;

    // Load tags
    std::unordered_set<std::string> tagSet = load_tags_from_code(pytag_file);
    if (tagSet.empty()) {
        std::cerr << "[ERROR] No tags loaded from: " << pytag_file << std::endl;
        return;
    }

    std::cout << "[INFO] Loaded " << tagSet.size() << " tags for scanning.\n";
    std::error_code ec;
    fs::recursive_directory_iterator it(directory, fs::directory_options::skip_permission_denied, ec), end;
    std::string ext = it->path().extension().string();
    std::transform(ext.begin(), ext.end(), ext.begin(), ::tolower);
    // Launch asynchronous tasks that return a JSON object.
    std::vector<std::future<json>> futures;

    while (it != end) {
        if (ec) {
            std::cerr << "[EXCEPTION] " << ec.message()
                << " -- Skipping: " << it->path() << std::endl;
            ec.clear();
            it.increment(ec);
            continue;
        }

        // Calculate the extension for the current file.
        std::string ext = it->path().extension().string();
        std::transform(ext.begin(), ext.end(), ext.begin(), ::tolower);

        std::cout << "[DEBUG] Checking: " << it->path()
            << " (is regular file: " << fs::is_regular_file(*it) << ")" << std::endl;

        // Only process regular files with .py extension.
        if (fs::is_regular_file(*it) && ext == ".py" || fs::is_regular_file(*it) && ext == ".cpp" || fs::is_regular_file(*it) && ext == ".h"
            || fs::is_regular_file(*it) && ext == ".cu" || fs::is_regular_file(*it) && ext == ".cuh") {
            std::cout << "[DEBUG] Found file: " << it->path() << std::endl;
            codeFiles.push_back(it->path().string());

            if (codeFiles.size() >= batchSize) {
                std::cout << "[DEBUG] Batch size reached: " << codeFiles.size() << " files." << std::endl;
                futures.push_back(std::async(std::launch::async,
                    [=, &tagSet](const std::vector<std::string>& files) -> json {
                        std::cout << "[DEBUG] Processing batch lambda with " << files.size() << " files." << std::endl;
                        return process_batch_return(files, tagSet);
                    },
                    codeFiles
                ));
                codeFiles.clear();
            }
        }
        it.increment(ec);
    }

    // Process any remaining files not making up a full batch.
    if (!codeFiles.empty()) {
        std::cout << "[DEBUG] Processing final batch of " << codeFiles.size() << " files." << std::endl;
        futures.push_back(std::async(std::launch::async,
            [](const std::vector<std::string>& files,
                const std::unordered_set<std::string>& tagSet) -> json {
                    std::cout << "[DEBUG] Processing final batch lambda with "
                        << files.size() << " files." << std::endl;
                    return process_batch_return(files, tagSet);
            },
            codeFiles, std::cref(tagSet)));
    }

    // Merge all results from the futures.
    for (auto& fut : futures) {
        json localResult = fut.get();
        std::cout << "[DEBUG] Batch result: " << localResult.dump(4) << std::endl;
        for (const auto& item : localResult.items()) {
            jsonData[item.key()] = item.value();
        }
    }

    // Write the merged JSON to a file.
    std::ofstream output("C:\\AI\\code_summary_results.json");
    if (!output) {
        std::cerr << "[ERROR] Failed to open JSON file for writing!" << std::endl;
    }
    else {
        output << jsonData.dump(4);
        std::cout << "[INFO] Successfully wrote to code_summary_results.json" << std::endl;
        output.close();
    }
}

void get_list_directory_full() {
    // Construct the command to call the Python script.
    // Adjust "python" to "python3" if needed based on your system configuration.
    const char* command = "python list_directory/list_directory_full.py";

    // Execute the command.
    int result = std::system(command);

    // Check the result and print an error if the command failed.
    if (result != 0) {
        std::cerr << "Error: Python script execution failed with code " << result << std::endl;
    }
}


BOOL APIENTRY DllMain(HMODULE hModule,
    DWORD  ul_reason_for_call,
    LPVOID lpReserved)
{
    switch (ul_reason_for_call)
    {
    case DLL_PROCESS_ATTACH:
    case DLL_THREAD_ATTACH:
    case DLL_THREAD_DETACH:
    case DLL_PROCESS_DETACH:
        break;
    }
    return TRUE;
}

extern "C" {
    __declspec(dllexport) void run(const char* dir, const char* tag_file, const char* code_file) {
        std::string directory = dir;
        get_list_directory_full();
        process_directory(directory, tag_file);
        process_directory_code(directory, code_file);
        std::cout << "[INFO] Program executed successfully." << std::endl;
    }
}


