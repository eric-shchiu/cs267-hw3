#pragma once

#include "kmer_t.hpp"
#include <upcxx/upcxx.hpp>
#include <vector>
#include <mutex>

struct HashMap {
    // Each rank owns a chunk of the hash table
    upcxx::global_ptr<kmer_pair> data;
    upcxx::global_ptr<int> used;
    size_t my_size;

    size_t total_size;
    int rank_n;
    int rank_me;

    HashMap(size_t total_size_);

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

    // Global-to-local mapping
    bool is_local(uint64_t slot);
    uint64_t local_index(uint64_t slot);
    uint64_t global_index_to_owner(uint64_t slot);
};

HashMap::HashMap(size_t total_size_) {
    total_size = total_size_;
    rank_n = upcxx::rank_n();
    rank_me = upcxx::rank_me();

    my_size = (total_size + rank_n - 1) / rank_n;
    data = upcxx::new_array<kmer_pair>(my_size);
    used = upcxx::new_array<int>(my_size);
    upcxx::barrier();
}

int HashMap::get_owner(const pkmer_t& key_kmer) {
    return key_kmer.hash() % rank_n;
}

bool HashMap::insert(const kmer_pair& kmer) {
    int owner = get_owner(kmer.kmer);
    if (owner == rank_me) {
        return insert_local(kmer);
    } else {
        return upcxx::rpc(owner, [](kmer_pair kmer_remote, upcxx::global_ptr<kmer_pair> d, upcxx::global_ptr<int> u, size_t size) {
            HashMap local_map(size);
            local_map.data = d;
            local_map.used = u;
            local_map.my_size = size;
            return local_map.insert_local(kmer_remote);
        }, kmer, data, used, my_size).wait();
    }
}

bool HashMap::find(const pkmer_t& key_kmer, kmer_pair& val_kmer) {
    int owner = get_owner(key_kmer);
    if (owner == rank_me) {
        return find_local(key_kmer, val_kmer);
    } else {
        auto result_pair = upcxx::rpc(owner, [](pkmer_t key, upcxx::global_ptr<kmer_pair> d, upcxx::global_ptr<int> u, size_t size) {
            HashMap local_map(size);
            local_map.data = d;
            local_map.used = u;
            local_map.my_size = size;
            kmer_pair result;
            bool found = local_map.find_local(key, result);
            return std::make_pair(found, result);
        }, key_kmer, data, used, my_size).wait();

        val_kmer = result_pair.second;
        return result_pair.first;
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
    int flag = upcxx::rget(used + slot).wait();
    if (flag != 0) return false;
    upcxx::rput(1, used + slot).wait();
    return true;
}
