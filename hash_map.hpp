#pragma once

#include "kmer_t.hpp"
#include <upcxx/upcxx.hpp>
#include <vector>
#include <mutex>

struct HashMap {
    // Each rank owns a chunk of the hash table
    upcxx::global_ptr<kmer_pair> data;
    upcxx::global_ptr<int> used;
    size_t my_size = 0;

    size_t total_size = 0;
    int rank_n = 0;
    int rank_me = 0;

    // Atomic domain for the 'used' array
    upcxx::atomic_domain<int>* ad_used = nullptr;

    // Static pointer to the instance on this rank (for RPCs)
    static HashMap* local_instance;

    HashMap() = default;

    // Make HashMap non-copyable because of the raw pointer ad_used
    HashMap(const HashMap&) = delete;
    HashMap& operator=(const HashMap&) = delete;
    // Define move semantics if needed, or keep it simple and non-movable too for now

    ~HashMap() {
        // Clean up atomic domain
        // Ensure this runs *before* upcxx::finalize
        if (ad_used != nullptr) {
            // Ensure all operations involving the domain are finished
            upcxx::barrier(); // Crucial barrier before destroying domain
            delete ad_used;
            ad_used = nullptr;
        }
        // Data/used memory is freed automatically by UPC++ on finalize
    }

    void init(size_t total_size_);

    // Hash function to decide owner
    int get_owner(const pkmer_t& key_kmer);

    // Distributed insert and find
    bool insert(const kmer_pair& kmer);
    bool find(const pkmer_t& key_kmer, kmer_pair& val_kmer);

    // Local operations for owned slots only
    bool insert_local(const kmer_pair& kmer);
    bool find_local(const pkmer_t& key_kmer, kmer_pair& val_kmer);

    // Slot operations (local only)
    void write_slot(uint64_t slot, const kmer_pair& kmer);
    kmer_pair read_slot(uint64_t slot);
    bool request_slot(uint64_t slot);
    bool slot_used(uint64_t slot);
};

// Define the static member outside the class definition
HashMap* HashMap::local_instance = nullptr;

void HashMap::init(size_t total_size_) {
    total_size = total_size_;
    rank_n = upcxx::rank_n();
    rank_me = upcxx::rank_me();

    my_size = (total_size + rank_n - 1) / rank_n;       // Round up (the type of my_size is size_t)
    data = upcxx::new_array<kmer_pair>(my_size);        // shared memory
    used = upcxx::new_array<int>(my_size);              // shared memory

    // *** Crucial: Initialize 'used' array to 0 ***
    // Simplest way: rput from rank 0, or each rank initializes its own
    // Let's do it locally and ensure visibility
    int* local_used = used.local();
    std::fill(local_used, local_used + my_size, 0);
    // Ensure initialization is visible before proceeding
    upcxx::barrier();


    // *** Create atomic domain for 'used' array ***
    // Use compare_swap operation, operate on the world team
    // Ensure ad_used is null before creating
    if (ad_used != nullptr) delete ad_used;
    // Team argument upcxx::world() means domain covers pointers on all ranks
    ad_used = new upcxx::atomic_domain<int>({}, upcxx::world());

    // Set the static pointer for this rank's instance
    HashMap::local_instance = this;

    // Barrier to ensure all ranks have initialized and set local_instance
    upcxx::barrier();
}

int HashMap::get_owner(const pkmer_t& key_kmer) {
    return key_kmer.hash() % rank_n;                    // assign kmer to different processor according to hash value (uniform distribution)
}

bool HashMap::insert(const kmer_pair& kmer) {
    int owner = get_owner(kmer.kmer);
    if (owner == rank_me) {
        return insert_local(kmer);
    } else {
        // Execute remotely using RPC and the static instance pointer
        return upcxx::rpc(owner,
            // Lambda captures the kmer to be inserted
            [](const kmer_pair& kmer_remote) {
                // On the target rank 'owner', access its local HashMap instance
                assert(HashMap::local_instance != nullptr); // Should be set during init
                return HashMap::local_instance->insert_local(kmer_remote);
            },
            kmer // Pass kmer as argument to lambda
        ).wait(); // Wait for the remote insertion to complete and return status
    }
}

bool HashMap::find(const pkmer_t& key_kmer, kmer_pair& val_kmer) {
    int owner = get_owner(key_kmer);
    if (owner == rank_me) {
        return find_local(key_kmer, val_kmer);
    } else {
        // Execute remotely using RPC
        // The result needs to be the pair {found_status, resulting_kmer}
        // Define a struct or use std::pair to return both
        using FindResult = std::pair<bool, kmer_pair>;

        FindResult result = upcxx::rpc(owner,
            // Lambda captures the key kmer
            [](const pkmer_t& key_kmer_remote) {
                 assert(HashMap::local_instance != nullptr);
                 kmer_pair result_kmer; // Local variable on target rank to store result
                 bool found = HashMap::local_instance->find_local(key_kmer_remote, result_kmer);
                 return std::make_pair(found, result_kmer);
            },
            key_kmer // Pass key_kmer as argument
        ).wait(); // Wait for the remote find to complete

        // Unpack the result
        val_kmer = result.second;
        return result.first;
    }
}

bool HashMap::insert_local(const kmer_pair& kmer) {
    uint64_t hash = kmer.hash();
    uint64_t probe = 0;
    while (probe < my_size) {
        uint64_t slot = (hash + probe++) % my_size;
        if (request_slot(slot)) {
            write_slot(slot, kmer);
            return true;
        }
    }
    return false;
}

bool HashMap::find_local(const pkmer_t& key_kmer, kmer_pair& val_kmer) {
    uint64_t hash = key_kmer.hash();
    uint64_t probe = 0;
    while (probe < my_size) {
        uint64_t slot = (hash + probe++) % my_size;
        if (slot_used(slot)) {
            kmer_pair k = read_slot(slot);
            if (k.kmer == key_kmer) {
                val_kmer = k;
                return true;
            }
        }
    }
    return false;
}

bool HashMap::slot_used(uint64_t slot) {
    return upcxx::rget(used + slot).wait() != 0;
}

void HashMap::write_slot(uint64_t slot, const kmer_pair& kmer) {
    upcxx::rput(kmer, data + slot).wait();
}

kmer_pair HashMap::read_slot(uint64_t slot) {
    return upcxx::rget(data + slot).wait();
}

bool HashMap::request_slot(uint64_t slot) {
    assert(ad_used != nullptr); // Atomic domain must be initialized
    const int increment = 1;

    // Atomically add 'increment' (1) to the value at 'used + slot'.
    // The operation returns the value that was at the location *before* the add.
    int old_value = ad_used->fetch_add(
        used + slot,          // Target global pointer
        increment,            // Value to add (1)
        std::memory_order_acq_rel // Memory order
    ).wait();                 // Wait for the operation and get the old value

    // If the old value was 0, it means we successfully transitioned it from 0 to 1.
    // We have successfully claimed the slot.
    if (old_value == 0) {
        return true; // Success!
    } else {
        // The old value was not 0, meaning someone else had already claimed it
        // (or potentially incremented it further, though ideally it should only be 0 or 1).
        // We did *not* claim the slot.
        //
        // Optional: We could potentially try to "undo" our increment if we failed,
        // by doing a fetch_add of -1, but that adds complexity and potential races
        // if multiple processes fail and try to undo. It's usually simpler
        // just to accept that the slot's value might now be > 1 if multiple
        // processes raced and failed after the first success. The key is that
        // only the *first* one (who saw old_value == 0) gets 'true'.
        return false; // Failure
    }
}