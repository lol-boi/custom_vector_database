#include "vectordb.h"
#include <iostream>
#include <string>
#include <vector>
#include <stdexcept>

// Helper function to parse a comma-separated vector string
std::vector<float> parseVector(const std::string& s, int expectedDim) {
    std::vector<float> vec;
    std::stringstream ss(s);
    std::string item;
    while (std::getline(ss, item, ',')) {
        try {
            vec.push_back(std::stof(item));
        } catch (...) {
            throw std::runtime_error("Invalid vector format. Must be comma-separated floats.");
        }
    }
    if (vec.size() != (size_t)expectedDim) {
        throw std::runtime_error("Vector dimension mismatch. Expected " + std::to_string(expectedDim) + " got " + std::to_string(vec.size()));
    }
    return vec;
}

// Helper to print usage instructions
void printUsage(const std::string& progName) {
    std::cerr << "Usage: " << progName << " <db_path> <command> [args]" << std::endl;
    std::cerr << "Commands:" << std::endl;
    std::cerr << "  init <dimension>                  - Initialize a new vector database." << std::endl;
    std::cerr << "  add <vector> <metadata_json>      - Add a new vector. Vector is '1.0,2.0,3.0'. Metadata is '{\"key\": \"val\"}'." << std::endl;
    std::cerr << "  get <id>                          - Get a vector and its metadata by ID." << std::endl;
    std::cerr << "  update <id> <vector> <metadata>   - Update a vector (requires rebuild)." << std::endl;
    std::cerr << "  delete <id>                       - Delete a vector (requires rebuild)." << std::endl;
    std::cerr << "  rebuild                         - Rebuild the HNSW index (REQUIRED after add/update/delete)." << std::endl;
    std::cerr << "  search <k> <query_vector>         - Search for k-nearest neighbors." << std::endl;
    std::cerr << std::endl;
}

int main(int argc, char** argv) {
    if (argc < 3) {
        printUsage(argv[0]);
        return 1;
    }

    std::string dbPath = argv[1];
    std::string command = argv[2];
    VectorDB db(dbPath);

    try {
        // --- init ---
        if (command == "init") {
            if (argc != 4) {
                std::cerr << "Usage: " << argv[0] << " " << dbPath << " init <dimension>" << std::endl;
                return 1;
            }
            int dim = std::stoi(argv[3]);
            db.init(dim);
            std::cout << "Database initialized at '" << dbPath << "' with dimension " << dim << std::endl;
        } 
        // --- add ---
        else if (command == "add") {
            if (argc != 5) {
                std::cerr << "Usage: " << argv[0] << " " << dbPath << " add <vector> <metadata_json>" << std::endl;
                return 1;
            }
            db.load(); // Load existing data first
            std::vector<float> vec = parseVector(argv[3], db.getDimensions());
            json metadata = json::parse(argv[4]);
            long long id = db.addVector(vec, metadata);
            db.save(); // Save after adding
            std::cout << "Vector added with ID: " << id << ". Run 'rebuild' to index." << std::endl;
        } 
        // --- get ---
        else if (command == "get") {
            if (argc != 4) {
                std::cerr << "Usage: " << argv[0] << " " << dbPath << " get <id>" << std::endl;
                return 1;
            }
            db.load();
            long long id = std::stoll(argv[3]);
            auto result = db.getVector(id);
            if (result.second) {
                std::cout << "ID: " << result.first.id << std::endl;
                std::cout << "Metadata: " << result.first.metadata.dump(2) << std::endl;
                std::cout << "Vector: [";
                for (size_t i = 0; i < result.first.vec.size(); ++i) {
                    std::cout << result.first.vec[i] << (i == result.first.vec.size() - 1 ? "" : ", ");
                }
                std::cout << "]" << std::endl;
            } else {
                std::cerr << "Error: Vector with ID " << id << " not found." << std::endl;
            }
        } 
        // --- search ---
        else if (command == "search") {
            if (argc != 5) {
                std::cerr << "Usage: " << argv[0] << " " << dbPath << " search <k> <query_vector>" << std::endl;
                return 1;
            }
            db.load();
            int k = std::stoi(argv[3]);
            
            // --- THIS IS THE FIX ---
            // The third argument was a copy-paste error and has been removed.
            std::vector<float> query = parseVector(argv[4], db.getDimensions());
            // --- END FIX ---

            auto results = db.search(query, k);

            std::cout << "Search results (ID, Distance):" << std::endl;
            if (results.empty()) {
                std::cout << "No results found. Have you run 'rebuild'?" << std::endl;
            }
            for (const auto& pair : results) {
                // We add std::sqrt here because the HNSW lib returns L2 SQUARED for performance
                std::cout << "- ID: " << pair.first << ", Dist: " << std::sqrt(pair.second) << std::endl;
            }
        }
        // --- rebuild ---
        else if (command == "rebuild") {
            db.load();
            std::cout << "Rebuilding index..." << std::endl;
            db.rebuildIndex();
            std::cout << "Index rebuild complete." << std::endl;
            // Note: We don't save the index, as it's purely in-memory.
            // A production DB would save the index binary to indexFilePath
        }
        // --- delete ---
        else if (command == "delete") {
             if (argc != 4) {
                std::cerr << "Usage: " << argv[0] << " " << dbPath << " delete <id>" << std::endl;
                return 1;
            }
            db.load();
            long long id = std::stoll(argv[3]);
            if (db.deleteVector(id)) {
                db.save();
                std::cout << "Vector " << id << " deleted. Run 'rebuild' to update index." << std::endl;
            } else {
                std::cerr << "Error: Vector with ID " << id << " not found." << std::endl;
            }
        }
        // --- update ---
        else if (command == "update") {
            if (argc != 6) {
                std::cerr << "Usage: " << argv[0] << " " << dbPath << " update <id> <vector> <metadata>" << std::endl;
                return 1;
            }
            db.load();
            long long id = std::stoll(argv[3]);
            std::vector<float> vec = parseVector(argv[4], db.getDimensions());
            json metadata = json::parse(argv[5]);
            if (db.updateVector(id, vec, metadata)) {
                db.save();
                std::cout << "Vector " << id << " updated. Run 'rebuild' to update index." << std::endl;
            } else {
                 std::cerr << "Error: Vector with ID " << id << " not found." << std::endl;
            }
        }
         else {
            std::cerr << "Unknown command: " << command << std::endl;
            printUsage(argv[0]);
            return 1;
        }

    } catch (const std::exception& e) {
        std::cerr << "Error: " << e.what() << std::endl;
        return 1;
    }

    return 0;
}

