
#include "vectordb.h"
#include <iostream>
#include <cassert>     // For our simple tests
#include <vector>
#include <cmath>       // For std::abs
#include <cstdio>      // For std::remove (to clean up)
#include <filesystem>  // For checking file existence

// Helper for float comparison
bool approx_equal(float a, float b) {
    return std::abs(a - b) < 1e-5;
}

// Helper to clean up test files
void cleanup(const std::string& path) {
    std::remove((path + ".json").c_str());
    // Note: .hnsw file is not saved by our current code, but good to have
    std::remove((path + ".hnsw").c_str()); 
}

void run_test(const std::string& test_name, std::function<void()> test_func) {
    std::cout << "Running test: " << test_name << "..." << std::endl;
    test_func();
    std::cout << "... " << test_name << " PASSED." << std::endl;
}

int main() {
    const std::string test_db_path = "./test_db";
    
    // --- Initial Cleanup ---
    cleanup(test_db_path);

    // --- Test 1: Init and Load ---
    run_test("Init and Load", [&]() {
        {
            VectorDB db(test_db_path);
            db.init(2); // Initialize with dimension 2
            assert(db.getDimensions() == 2);
            std::cout << "  - init() ok." << std::endl;
        } // db goes out of scope

        {
            VectorDB db2(test_db_path);
            db2.load(); // Load the file we just created
            assert(db2.getDimensions() == 2);
            auto res = db2.getVector(1);
            assert(res.second == false); // Should be empty
            std::cout << "  - load() ok." << std::endl;
        }
    });

    // --- Test 2: Add, Save, and Get ---
    run_test("Add, Save, and Get", [&]() {
        long long id1, id2;
        {
            VectorDB db(test_db_path);
            db.load(); // Load the empty 2-dim db
            
            id1 = db.addVector({1.0f, 1.1f}, {{"name", "vec1"}});
            id2 = db.addVector({10.0f, 10.1f}, {{"name", "vec2"}});
            
            assert(id1 == 1);
            assert(id2 == 2);
            db.save(); // Save the two new vectors
            std::cout << "  - addVector() and save() ok." << std::endl;
        }

        {
            VectorDB db2(test_db_path);
            db2.load(); // Load db with 2 vectors
            assert(db2.getDimensions() == 2);

            auto res1 = db2.getVector(id1);
            assert(res1.second == true);
            assert(res1.first.metadata["name"] == "vec1");
            assert(approx_equal(res1.first.vec[0], 1.0f));

            auto res2 = db2.getVector(id2);
            assert(res2.second == true);
            assert(res2.first.metadata["name"] == "vec2");
            assert(approx_equal(res2.first.vec[0], 10.0f));

            auto res3 = db2.getVector(999);
            assert(res3.second == false);
            std::cout << "  - getVector() ok." << std::endl;
        }
    });

    // --- Test 3: Search (load() includes rebuild) ---
    run_test("Search", [&]() {
        VectorDB db(test_db_path);
        db.load(); // This loads AND rebuilds the index with 2 vectors

        auto results1 = db.search({1.0f, 1.0f}, 1);
        assert(results1.size() == 1);
        assert(results1[0].first == 1); // ID 1 is closest
        std::cout << "  - Search for vec1 ok." << std::endl;

        auto results2 = db.search({11.0f, 11.0f}, 1);
        assert(results2.size() == 1);
        assert(results2[0].first == 2); // ID 2 is closest
        std::cout << "  - Search for vec2 ok." << std::endl;
    });


    // --- Test 4: Delete, Rebuild, and Search ---
    run_test("Delete and Rebuild", [&]() {
        VectorDB db(test_db_path);
        db.load();
        
        bool deleted = db.deleteVector(1); // Delete vec1
        assert(deleted == true);
        
        // Search *before* rebuild (index is stale)
        auto old_results = db.search({1.0f, 1.0f}, 1);
        assert(old_results[0].first == 1); // Still finds old ID 1
        std::cout << "  - Search on stale index ok." << std::endl;

        db.rebuildIndex(); // Rebuild with only vec2
        
        // Search *after* rebuild
        auto new_results = db.search({1.0f, 1.0f}, 1);
        assert(new_results.size() == 1);
        assert(new_results[0].first == 2); // Now finds ID 2 as closest
        std::cout << "  - Delete and rebuild ok." << std::endl;
    });

    // --- Test 5: Update, Rebuild, and Search ---
    run_test("Update and Rebuild", [&]() {
        VectorDB db(test_db_path);
        db.load(); // Has only vec2 (ID 2)
        
        bool updated = db.updateVector(2, {20.0f, 20.0f}, {{"name", "vec2_updated"}});
        assert(updated == true);

        db.rebuildIndex(); // Rebuild with vec2 at new position

        // Search near old position
        auto results1 = db.search({10.0f, 10.0f}, 1);
        assert(results1.empty() || results1[0].first != 2); // Should not find ID 2
        std::cout << "  - Search old position ok." << std::endl;
        
        // Search near new position
        auto results2 = db.search({20.1f, 20.1f}, 1);
        assert(results2.size() == 1);
        assert(results2[0].first == 2);
        std::cout << "  - Search new position ok." << std::endl;

        // Check metadata update
        auto res = db.getVector(2);
        assert(res.first.metadata["name"] == "vec2_updated");
        std::cout << "  - Metadata update ok." << std::endl;
    });


    std::cout << "\n---------------------" << std::endl;
    std::cout << "ALL TESTS PASSED!" << std::endl;
    std::cout << "---------------------" << std::endl;
    
    // --- Final Cleanup ---
    cleanup(test_db_path);
    return 0;
}
