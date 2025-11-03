
#ifndef HNSW_H
#define HNSW_H

#include <iostream>
#include <vector>
#include <queue>
#include <map>
#include <set>
#include <mutex>
#include <thread>
#include <random>
#include <stdexcept>
#include <cmath> // For std::sqrt

/*
This is a C++ implementation of HNSW,
based on the paper "Efficient and robust approximate nearest neighbor search using Hierarchical Navigable Small World graphs" (Yu. A. Malkov, D. A. Yashunin).
The implementation is from https://github.com/xinranhe/HNSW
It has been slightly modified to fix compilation errors and C++ correctness.
*/

class HNSW {
public:
    // M, M_max, M_max0, ef_construction, L, ml
    HNSW(int dim, int max_elements, int M = 16, int M_max0 = 32, int ef_construction = 200) :
        dim_(dim), max_elements_(max_elements), M_(M), M_max0_(M_max0), ef_construction_(ef_construction) {
        
        // ml = 1/log(M)
        ml = 1.0 / log(1.0 * M_); 
        L_ = 0; // Current max layer
        
        // Initialize layers. We reserve space.
        // We use max_elements_ as an estimate.
        nodes_.reserve(max_elements_);
        
        // Initialize the enter point
        enter_point_ = -1;

        // Default to L2 distance
        dist_func_ = L2Sqr;
    }

    // L2 Distance function
    static float L2Sqr(const float *a, const float *b, int dim) {
        float sum = 0;
        for (int i = 0; i < dim; ++i) {
            float diff = a[i] - b[i];
            sum += diff * diff;
        }
        return sum;
    }

    void addPoint(const float* p, int label) {
        std::unique_lock<std::mutex> lock(mutex_);
        
        int id = nodes_.size();
        nodes_.push_back(Node(p, label, dim_));
        
        int l = getRandomLayer();
        if (l > L_) L_ = l;

        int ep = enter_point_;

        if (ep == -1) {
            enter_point_ = id;
            return;
        }

        for (int lc = L_; lc > l; --lc) {
            ep = searchLayer(p, ep, 1, lc).top().second;
        }

        for (int lc = std::min(l, L_); lc >= 0; --lc) {
            int M_max = (lc == 0) ? M_max0_ : M_;
            std::priority_queue<std::pair<float, int>> W = searchLayer(p, ep, ef_construction_, lc);
            
            // This is SELECT-NEIGHBORS-SIMPLE from the paper
            std::priority_queue<std::pair<float, int>> neighbors;
            while (neighbors.size() < (unsigned int)M_ && !W.empty()) {
                neighbors.push(W.top());
                W.pop();
            }

            // Add connections
            int ep_label = ep; // Save ep before it's potentially updated
            while (!neighbors.empty()) {
                int neighbor_id = neighbors.top().second;
                neighbors.pop();
                
                nodes_[id].addNeighbor(lc, neighbor_id);
                nodes_[neighbor_id].addNeighbor(lc, id);

                // Check for over-connection (pruning)
                if (nodes_[neighbor_id].friends[lc].size() > (unsigned int)M_max) {
                    pruneConnections(neighbor_id, lc, M_max);
                }
            }
            ep = ep_label; // Use the original ep for the next layer down
        }
    }


    std::priority_queue<std::pair<float, int>> searchKnn(const float* q, int k) {
        std::unique_lock<std::mutex> lock(mutex_);
        
        int ep = enter_point_;
        if (ep == -1) {
            return std::priority_queue<std::pair<float, int>>();
        }

        for (int lc = L_; lc >= 1; --lc) {
            ep = searchLayer(q, ep, 1, lc).top().second;
        }
        
        // W is a max-heap of (distance, internal_id) for the k-closest items
        std::priority_queue<std::pair<float, int>> W = searchLayer(q, ep, k, 0);

        // --- THIS IS THE FIX ---
        // We must convert internal_id to external label.
        std::priority_queue<std::pair<float, int>> results;
        while (!W.empty()) {
            std::pair<float, int> top = W.top(); // (dist, internal_id)
            W.pop();
            
            // Find the external label and push (dist, external_label)
            results.push(std::make_pair(top.first, nodes_[top.second].label));
        }
        // --- END FIX ---
        
        return results; // Return the new heap with external labels
    }


private:
    int dim_;
    int max_elements_;
    int M_;
    int M_max0_;
    int ef_construction_;
    double ml;
    int L_; // Max layer
    int enter_point_;

    std::mutex mutex_;
    std::default_random_engine generator_;

    // Distance function pointer
    float (*dist_func_)(const float*, const float*, int);

    struct Node {
        std::vector<float> data;
        int label;
        std::vector<std::vector<int>> friends; // friends[layer][neighbor_id]

        Node(const float* p, int label, int dim) : label(label) {
            data.resize(dim);
            std::copy(p, p + dim, data.begin());
        }

        void addNeighbor(int layer, int neighbor_id) {
            if (layer >= (int)friends.size()) {
                friends.resize(layer + 1);
            }
            friends[layer].push_back(neighbor_id);
        }
    };

    std::vector<Node> nodes_; // All nodes

    float dist(const float* q, int node_id, int layer) {
        return dist_func_(q, nodes_[node_id].data.data(), dim_);
    }

    int getRandomLayer() {
        std::uniform_real_distribution<double> distribution(0.0, 1.0);
        int l = 0;
        while (distribution(generator_) < ml && l < 16) { // Cap at 16 layers
            l++;
        }
        return l;
    }

    void pruneConnections(int node_id, int layer, int M_max) {
        // --- THIS IS THE FIX ---
        // We must use a min-heap (std::greater) to find the *closest* neighbors to keep.
        // The old code used a max-heap, which kept the *farthest* neighbors.
        std::priority_queue<std::pair<float, int>, std::vector<std::pair<float, int>>, std::greater<std::pair<float, int>>> connections;
        // --- END FIX ---
        
        for (int neighbor_id : nodes_[node_id].friends[layer]) {
            // Use dist_func_ directly for node-to-node distance
            connections.push(std::make_pair(dist_func_(nodes_[node_id].data.data(), nodes_[neighbor_id].data.data(), dim_), neighbor_id));
        }

        nodes_[node_id].friends[layer].clear();
        while (nodes_[node_id].friends[layer].size() < (unsigned int)M_max && !connections.empty()) {
            nodes_[node_id].friends[layer].push_back(connections.top().second);
            connections.pop();
        }
    }

    std::priority_queue<std::pair<float, int>> searchLayer(const float* q, int ep, int ef, int l) {
        std::priority_queue<std::pair<float, int>> W; // min-heap of (dist, id)
        std::priority_queue<std::pair<float, int>, std::vector<std::pair<float, int>>, std::greater<std::pair<float, int>>> C; // max-heap of (dist, id)
        
        std::set<int> visited;

        // --- These are the corrected lines ---
        C.push(std::make_pair(dist(q, ep, l), ep));
        W.push(std::make_pair(dist(q, ep, l), ep));
        // --- End corrected lines ---

        visited.insert(ep);

        while (!C.empty()) {
            int c = C.top().second;
            C.pop();

            // --- Corrected line ---
            if (c == -1 || W.top().first < dist(q, c, l)) {
                break;
            }
            // --- End corrected line ---

            // --- THIS IS THE FIX ---
            // Check if the node 'c' has a friends list for layer 'l' before accessing it
            if (nodes_[c].friends.size() > (unsigned int)l) {
                for (int e : nodes_[c].friends[l]) {
                    if (visited.find(e) == visited.end()) {
                        visited.insert(e);
                        // --- Corrected lines ---
                        float d_e = dist(q, e, l);
                        if (d_e < W.top().first || W.size() < (unsigned int)ef) {
                            W.push(std::make_pair(d_e, e));
                            C.push(std::make_pair(d_e, e));
                            // --- End corrected lines ---
                            if (W.size() > (unsigned int)ef) {
                                W.pop();
                            }
                        }
                    }
                }
            }
            // --- END FIX ---
        }
        return W;
    }
};

#endif // HNSW_H

