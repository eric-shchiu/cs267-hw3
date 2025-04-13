#pragma once

#include "kmer_t.hpp"
#include <upcxx/upcxx.hpp>

struct HashMap {
    // Vectors of gptrs to each rank's data and used arrays.
    std::vector<upcxx::global_ptr<kmer_pair>> data_ptrs;
    std::vector<upcxx::global_ptr<int>> used_ptrs;
    
    // Atomic domain for used array.
    upcxx::atomic_domain<int> ad;
    size_t my_size;
    
    // Number of slots owned by each rank
    size_t local_size;
    size_t size() const noexcept;
    HashMap(size_t size);
    
    // Most important functions: insert and retrieve
    // k-mers from the hash table.
    bool insert(const kmer_pair& kmer);
    bool find(const pkmer_t& key_kmer, kmer_pair& val_kmer);
   
    // Helper functions
    // Write and read to a logical data slot in the table.
    void write_slot(uint64_t slot, const kmer_pair& kmer);
    kmer_pair read_slot(uint64_t slot);
    
    // Request a slot or check if it's already used.
    bool request_slot(uint64_t slot);
    bool slot_used(uint64_t slot);
   };

HashMap::HashMap(size_t size) {
    my_size = size;
    
    // Resize to number of ranks.
    int num_ranks = upcxx::rank_n();
    data_ptrs.resize(num_ranks);
    used_ptrs.resize(num_ranks);
    
    // Number of slots owned by each rank.
    local_size = (size + num_ranks - 1) / num_ranks; // ceiling division
    
    // // Allocate data and used arrays for each rank.
    kmer_pair* local_data = new kmer_pair[local_size];
    int* local_used = new int[local_size]();  // Initialize to zeros
    
    // Register the local arrays with UPC++
    upcxx::global_ptr<kmer_pair> my_data_ptr = upcxx::new_array<kmer_pair>(local_size);
    upcxx::global_ptr<int> my_used_ptr = upcxx::new_array<int>(local_size);
    
    // Initialize the used array to all zeros
    for (size_t i = 0; i < local_size; i++) {
        local_used[i] = 0;
    }
    
    // Copy local arrays to UPC++ allocated memory
    for (size_t i = 0; i < local_size; i++) {
        upcxx::rput(local_data[i], my_data_ptr + i).wait();
        upcxx::rput(local_used[i], my_used_ptr + i).wait();
    }

     // Clean up local arrays
    delete[] local_data;
    delete[] local_used;
 
    // Initialize atomic domain for atomic operations
    ad = upcxx::atomic_domain<int>({upcxx::atomic_op::compare_exchange, 
                                    upcxx::atomic_op::fetch_add});
    
    // Broadcast the global pointers to all ranks.
    for (int i = 0; i < num_ranks; i++) {
        data_ptrs[i] = upcxx::broadcast(my_data_ptr, i).wait();
        used_ptrs[i] = upcxx::broadcast(my_used_ptr, i).wait();
    }
}

bool HashMap::insert(const kmer_pair& kmer) {
    uint64_t hash = kmer.hash();
    uint64_t probe = 0;
    bool success = false;
    do {
        uint64_t slot = (hash + probe++) % size();
        success = request_slot(slot);
        if (success) {
            write_slot(slot, kmer);
        }
    } while (!success && probe < size());
    return success;
   }
  

bool HashMap::find(const pkmer_t& key_kmer, kmer_pair& val_kmer) {
    uint64_t hash = key_kmer.hash();
    uint64_t probe = 0;
    bool success = false;
    do {
        uint64_t slot = (hash + probe++) % size();
        if (slot_used(slot)) {
            val_kmer = read_slot(slot);
            if (val_kmer.kmer == key_kmer) {
                success = true;
            }
        }
    } while (!success && probe < size());
    return success;
}

/*
For a slot, we need to find the rank that owns that slot, and the local index of that slot.
Then use rget/rput to read/write the used value for the slot.
*/

bool HashMap::slot_used(uint64_t slot) {
    // Find the rank that owns the slot and the local index of that slot.
    int owner_rank = slot / local_size;
    uint64_t local_idx = slot % local_size;
    
    // Use rget to check if the slot is used.
    int used_value = upcxx::rget(used_ptrs[owner_rank] + local_idx).wait();
    
    // Return true if the slot is used (!=0), false otherwise.
    return used_value != 0;
}

void HashMap::write_slot(uint64_t slot, const kmer_pair& kmer) {
    // Find the rank that owns the slot and the local index of that slot.
    int owner_rank = slot / local_size;
    uint64_t local_idx = slot % local_size;
    
    // Use rput to write the kmer to the slot.
    upcxx::rput(kmer, data_ptrs[owner_rank] + local_idx).wait();
}
   
kmer_pair HashMap::read_slot(uint64_t slot) {
    // Find the rank that owns the slot and the local index of that slot.
    int owner_rank = slot / local_size;
    uint64_t local_idx = slot % local_size;
    
    // Use rget to read the kmer from the slot.
    return upcxx::rget(data_ptrs[owner_rank] + local_idx).wait();
}

bool HashMap::request_slot(uint64_t slot) {
    // Find the rank that owns the slot and the local index of that slot.
    int owner_rank = slot / local_size;
    uint64_t local_idx = slot % local_size;
    
    // Use atomic fetch_add to request the slot.
    // If the slot was unused (0), fetch_add will return 0 and set it to 1
    int old_value = ad.fetch_add(used_ptrs[owner_rank] + local_idx, 1, 
                                std::memory_order_relaxed).wait();
    
    // If old_value was 0, the slot was available and now we have it
    if (old_value != 0) {
        // Slot was already in use, so we need to undo our increment
        ad.fetch_add(used_ptrs[owner_rank] + local_idx, -1, 
                    std::memory_order_relaxed).wait();
        return false;
    }
    
    return true;
}

size_t HashMap::size() const noexcept { return my_size; }
