#ifndef VECTORDB_H
#define VECTORDB_H

#include <iostream>
#include <fstream>
#include <string>
#include <vector>
#include <map>
#include <stdexcept>
#include <sstream>
#include <memory>

// The HNSW library header
#include "hnsw.h" 
// The JSON library header
#include "json.hpp"

// Use the nlohmann::json library
using json = nlohmann::json;

// Holds our metadata and the raw vector data
struct VectorData {
    long long id;
    std::vector<float> vec;
    json metadata;
};

class VectorDB {
public:
    VectorDB(const std::string& dbPath);
    ~VectorDB();

    void init(int dim);
    long long addVector(const std::vector<float>& vec, const json& metadata);
    std::pair<VectorData, bool> getVector(long long id);
    bool updateVector(long long id, const std::vector<float>& vec, const json& metadata);
    bool deleteVector(long long id);

    void rebuildIndex();
    std::vector<std::pair<long long, float>> search(const std::vector<float>& query, int k);

    void save();
    void load();

    // Public getter for the dimension
    int getDimensions() const;

private:
    std::string dbPath;
    std::string dataFilePath;
    std::string indexFilePath; // We aren't using this yet, but good to have

    int dim; // Vector dimensionality
    long long nextId;
    std::map<long long, VectorData> vectors; // Stores all data
    
    // The HNSW index. We use a unique_ptr to manage its lifecycle
    // because we will be deleting and recreating it on rebuild.
    std::unique_ptr<HNSW> hnsw_index;

    // Helper to store raw pointers for HNSW
    // This is rebuilt by rebuildIndex()
    std::vector<float> raw_vector_data;
};

#endif // VECTORDB_H

