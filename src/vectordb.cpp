#include "vectordb.h"
#include <stdexcept>
#include <fstream>
#include <filesystem> // For checking file existence

// --- Constructor & Destructor ---

VectorDB::VectorDB(const std::string& dbPath) : 
    dbPath(dbPath),
    dataFilePath(dbPath + ".json"),
    indexFilePath(dbPath + ".hnsw"), // We don't use this yet, but good practice
    dim(0), 
    nextId(0) {
    // Constructor body. We call load() to populate the db.
}

VectorDB::~VectorDB() {
    // Destructor
}

// --- Public API ---

void VectorDB::init(int dimension) {
    if (std::filesystem::exists(dataFilePath)) {
        throw std::runtime_error("Database file already exists. Cannot initialize.");
    }
    this->dim = dimension;
    this->nextId = 1; // Start IDs at 1
    this->vectors.clear();
    
    // Create an empty index
    rebuildIndex(); 
    
    // Save the empty state
    save();
}

long long VectorDB::addVector(const std::vector<float>& vec, const json& metadata) {
    if (vec.size() != (size_t)this->dim) {
        throw std::runtime_error("Vector dimension mismatch.");
    }

    long long id = nextId++;
    VectorData data;
    data.id = id;
    data.vec = vec;
    data.metadata = metadata;
    
    vectors[id] = data;
    // Note: Does not rebuild index. User must call rebuild().
    return id;
}

std::pair<VectorData, bool> VectorDB::getVector(long long id) {
    if (vectors.find(id) != vectors.end()) {
        return {vectors[id], true};
    }
    return {{}, false};
}

bool VectorDB::updateVector(long long id, const std::vector<float>& vec, const json& metadata) {
    if (vectors.find(id) == vectors.end()) {
        return false; // Not found
    }
     if (vec.size() != (size_t)this->dim) {
        throw std::runtime_error("Vector dimension mismatch.");
    }
    
    vectors[id].vec = vec;
    vectors[id].metadata = metadata;
    return true;
}

bool VectorDB::deleteVector(long long id) {
    if (vectors.find(id) == vectors.end()) {
        return false; // Not found
    }
    vectors.erase(id);
    return true;
}

void VectorDB::rebuildIndex() {
    // 1. Prepare the raw data in the format HNSW needs
    // We need a single, flat array of floats
    raw_vector_data.clear();
    raw_vector_data.reserve(vectors.size() * dim);
    
    // We also need a map to get from the HNSW's internal index (0, 1, 2...)
    // back to our external ID (1, 10, 105...)
    // For this simple library, the internal label IS the index.
    
    int internal_index = 0;
    std::map<int, long long> internal_to_external_id;

    for (auto const& [id, data] : vectors) {
        raw_vector_data.insert(raw_vector_data.end(), data.vec.begin(), data.vec.end());
        internal_to_external_id[internal_index] = id;
        internal_index++;
    }

    // 2. Create a new, empty index
    int max_elements = std::max((int)vectors.size(), 1); // Ensure not zero
    hnsw_index = std::make_unique<HNSW>(dim, max_elements, 16, 200);

    // 3. Add all points to the index
    if (raw_vector_data.empty()) {
        std::cerr << "Warning: Rebuilding index with 0 vectors." << std::endl;
        return; // Nothing to index
    }

    for (int i = 0; i < (int)vectors.size(); ++i) {
        // Get the pointer to the start of the i-th vector
        float* p = raw_vector_data.data() + (i * dim);
        
        // Use the internal index (0, 1, 2...) as the label
        hnsw_index->addPoint(p, i);
    }
}

std::vector<std::pair<long long, float>> VectorDB::search(const std::vector<float>& query, int k) {
    if (!hnsw_index) {
        throw std::runtime_error("Index is not built. Run 'rebuild' first.");
    }
    if (query.size() != (size_t)dim) {
        throw std::runtime_error("Query vector dimension mismatch.");
    }

    auto result_queue = hnsw_index->searchKnn(query.data(), k);

    std::vector<std::pair<long long, float>> results;
    
    // The HNSW lib gives internal labels (0, 1, 2...)
    // We need to map them back to our external IDs (1, 10, 105...)
    // We can rebuild this map easily
    std::map<int, long long> internal_to_external_id;
    int i = 0;
    for (auto const& [id, data] : vectors) {
        internal_to_external_id[i++] = id;
    }

    while (!result_queue.empty()) {
        auto top = result_queue.top();
        result_queue.pop();
        
        float dist = top.first;
        int internal_id = top.second;

        if (internal_to_external_id.count(internal_id)) {
            long long external_id = internal_to_external_id[internal_id];
            results.push_back({external_id, dist});
        }
    }
    // The queue gives results in (farthest, ... , nearest) order
    std::reverse(results.begin(), results.end());
    return results;
}

void VectorDB::save() {
    json j;
    j["dim"] = this->dim;
    j["nextId"] = this->nextId;
    json& j_vectors = j["vectors"];
    
    for (auto const& [id, data] : vectors) {
        json j_vec;
        j_vec["id"] = data.id;
        j_vec["metadata"] = data.metadata;
        j_vec["vec"] = data.vec;
        j_vectors.push_back(j_vec);
    }

    std::ofstream o(dataFilePath);
    if (!o.is_open()) {
        throw std::runtime_error("Failed to open database file for writing: " + dataFilePath);
    }
    o << j.dump(2); // pretty print with 2 spaces
    o.close();
}

void VectorDB::load() {
    std::ifstream i(dataFilePath);
    if (!i.is_open()) {
        // This is not an error if the file just doesn't exist yet
        // std::cerr << "Warning: Database file not found. Starting fresh." << std::endl;
        return;
    }

    json j;
    try {
        i >> j;
    } catch (json::parse_error& e) {
        i.close();
        throw std::runtime_error("Failed to parse database file (JSON error): " + std::string(e.what()));
    }
    i.close();

    try {
        this->dim = j.at("dim").get<int>();
        this->nextId = j.at("nextId").get<long long>();
        
        this->vectors.clear();
        if (j.contains("vectors")) {
            for (const auto& j_vec : j["vectors"]) {
                VectorData data;
                data.id = j_vec.at("id").get<long long>();
                data.metadata = j_vec.at("metadata");
                data.vec = j_vec.at("vec").get<std::vector<float>>();
                
                vectors[data.id] = data;
            }
        }
    } catch (json::exception& e) {
        throw std::runtime_error("Database file is corrupted (missing fields): " + std::string(e.what()));
    }

    // After loading data, we MUST rebuild the in-memory index
    rebuildIndex();
}

int VectorDB::getDimensions() const {
    return this->dim;
}

