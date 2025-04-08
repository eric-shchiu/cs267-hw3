#include <chrono>
#include <cstddef>
#include <cstdio>
#include <cstdlib>
#include <list>
#include <numeric>
#include <set>
#include <upcxx/upcxx.hpp>
#include <vector>
#include <fstream>

#include "hash_map.hpp"
#include "kmer_t.hpp"
#include "read_kmers.hpp"
#include "butil.hpp"

int main(int argc, char** argv) {
    upcxx::init();

    if (argc < 2) {
        BUtil::print("usage: srun -N nodes -n ranks ./kmer_hash kmer_file [verbose|test [prefix]]\n");
        upcxx::finalize();
        exit(1);
    }

    std::string kmer_fname = std::string(argv[1]);
    std::string run_type = (argc >= 3) ? argv[2] : "";
    std::string test_prefix = (run_type == "test" && argc >= 4) ? argv[3] : "test";

    int ks = kmer_size(kmer_fname);
    if (ks != KMER_LEN) {
        throw std::runtime_error("Error: " + kmer_fname + " contains " + std::to_string(ks) +
                                 "-mers, while this binary is compiled for " +
                                 std::to_string(KMER_LEN) + "-mers.  Modify packing.hpp and recompile.");
    }

    size_t n_kmers = line_count(kmer_fname);
    size_t hash_table_size = n_kmers * 2; // Load factor 0.5
    HashMap hashmap(hash_table_size);

    if (run_type == "verbose") {
        BUtil::print("Rank %d: Initializing hash table of size %lu for %lu kmers.\n", 
                     upcxx::rank_me(), hash_table_size, n_kmers);
    }

    std::vector<kmer_pair> kmers = read_kmers(kmer_fname, upcxx::rank_n(), upcxx::rank_me());

    if (run_type == "verbose") {
        BUtil::print("Rank %d: Finished reading kmers.\n", upcxx::rank_me());
    }

    auto start = std::chrono::high_resolution_clock::now();

    std::vector<kmer_pair> start_nodes;
    for (auto& kmer : kmers) {
        bool success = hashmap.insert(kmer);
        if (!success) {
            throw std::runtime_error("Error: HashMap is full!");
        }
        if (kmer.backwardExt() == 'F') {
            start_nodes.push_back(kmer);
        }
    }

    auto end_insert = std::chrono::high_resolution_clock::now();
    upcxx::barrier();

    double insert_time = std::chrono::duration<double>(end_insert - start).count();
    if (run_type != "test") {
        BUtil::print("Rank %d: Finished inserting in %lf seconds\n", upcxx::rank_me(), insert_time);
    }

    upcxx::barrier();
    auto start_read = std::chrono::high_resolution_clock::now();

    std::list<std::list<kmer_pair>> contigs;
    for (const auto& start_kmer : start_nodes) {
        std::list<kmer_pair> contig;
        contig.push_back(start_kmer);
        while (contig.back().forwardExt() != 'F') {
            kmer_pair next;
            bool found = hashmap.find(contig.back().next_kmer(), next);
            if (!found) {
                throw std::runtime_error("Error: k-mer not found in hashmap.");
            }
            contig.push_back(next);
        }
        contigs.push_back(contig);
    }

    auto end_read = std::chrono::high_resolution_clock::now();
    upcxx::barrier();
    auto end = std::chrono::high_resolution_clock::now();

    std::chrono::duration<double> read = end_read - start_read;
    std::chrono::duration<double> insert = end_insert - start;
    std::chrono::duration<double> total = end - start;

    int numKmers = std::accumulate(
        contigs.begin(), contigs.end(), 0,
        [](int sum, const std::list<kmer_pair>& contig) { return sum + contig.size(); });

    if (run_type != "test") {
        BUtil::print("Rank %d: Assembled in %lf seconds total\n", upcxx::rank_me(), total.count());
    }

    if (run_type == "verbose") {
        printf("Rank %d reconstructed %lu contigs with %d nodes from %lu start nodes.\n"
               " (%lf read, %lf insert, %lf total)\n",
               upcxx::rank_me(), contigs.size(), numKmers, start_nodes.size(),
               read.count(), insert.count(), total.count());
    }

    if (run_type == "test") {
        std::ofstream fout(test_prefix + "_" + std::to_string(upcxx::rank_me()) + ".dat");
        for (const auto& contig : contigs) {
            fout << extract_contig(contig) << std::endl;
        }
        fout.close();
    }

    upcxx::finalize();
    return 0;
}

// #include <chrono>
// #include <cstddef>
// #include <cstdio>
// #include <cstdlib>
// #include <list>
// #include <numeric>
// #include <set>
// #include <upcxx/upcxx.hpp>
// #include <vector>

// #include "hash_map.hpp"
// #include "kmer_t.hpp"
// #include "read_kmers.hpp"

// #include "butil.hpp"

// int main(int argc, char** argv) {
//     upcxx::init();

//     if (argc < 2) {
//         BUtil::print("usage: srun -N nodes -n ranks ./kmer_hash kmer_file [verbose|test [prefix]]\n");
//         upcxx::finalize();
//         exit(1);
//     }

//     std::string kmer_fname = std::string(argv[1]);
//     std::string run_type = "";

//     if (argc >= 3) {
//         run_type = std::string(argv[2]);
//     }

//     std::string test_prefix = "test";
//     if (run_type == "test" && argc >= 4) {
//         test_prefix = std::string(argv[3]);
//     }

//     int ks = kmer_size(kmer_fname);

//     if (ks != KMER_LEN) {
//         throw std::runtime_error("Error: " + kmer_fname + " contains " + std::to_string(ks) +
//                                  "-mers, while this binary is compiled for " +
//                                  std::to_string(KMER_LEN) +
//                                  "-mers.  Modify packing.hpp and recompile.");
//     }

//     size_t n_kmers = line_count(kmer_fname);

//     // Load factor of 0.5
//     size_t hash_table_size = n_kmers * (1.0 / 0.5);
//     HashMap hashmap(hash_table_size);

//     if (run_type == "verbose") {
//         BUtil::print("Initializing hash table of size %d for %d kmers.\n", hash_table_size,
//                      n_kmers);
//     }

//     // read the k-mers dataset
//     std::vector<kmer_pair> kmers = read_kmers(kmer_fname, upcxx::rank_n(), upcxx::rank_me());

//     if (run_type == "verbose") {
//         BUtil::print("Finished reading kmers.\n");
//     }

//     auto start = std::chrono::high_resolution_clock::now();

//     // -------------------------Start:  Fill the hash table and record the starting node-------------------------
//     // create an empty list to save the start_nodes
//     std::vector<kmer_pair> start_nodes;
//     // fill the hash table
//     for (auto& kmer : kmers) {
//         bool success = hashmap.insert(kmer);
//         if (!success) {
//             throw std::runtime_error("Error: HashMap is full!");
//         }
        
//         // Record the starting node
//         if (kmer.backwardExt() == 'F') {
//             start_nodes.push_back(kmer);
//         }
//     }
//     // -------------------------End:  Fill the hash table and record the starting node-------------------------

//     auto end_insert = std::chrono::high_resolution_clock::now();
//     upcxx::barrier();

//     double insert_time = std::chrono::duration<double>(end_insert - start).count();
//     if (run_type != "test") {
//         BUtil::print("Finished inserting in %lf\n", insert_time);
//     }
//     upcxx::barrier();

//     auto start_read = std::chrono::high_resolution_clock::now();

//     // -------------------------Start: Build the Chain table-------------------------
//     // Iterate over contigs to build a chain table
//     std::list<std::list<kmer_pair>> contigs;
//     for (const auto& start_kmer : start_nodes) {
//         std::list<kmer_pair> contig;
//         contig.push_back(start_kmer);
//         while (contig.back().forwardExt() != 'F') {
//             // create a empty kmer_pair variable (this is the return value in .find function)
//             kmer_pair kmer;
//             bool success = hashmap.find(contig.back().next_kmer(), kmer);
//             if (!success) {
//                 throw std::runtime_error("Error: k-mer not found in hashmap.");
//             }
//             contig.push_back(kmer);
//         }
//         contigs.push_back(contig);
//     }
//     // -------------------------End: Build the Chain table-------------------------

//     auto end_read = std::chrono::high_resolution_clock::now();
//     upcxx::barrier();
//     auto end = std::chrono::high_resolution_clock::now();

//     std::chrono::duration<double> read = end_read - start_read;
//     std::chrono::duration<double> insert = end_insert - start;
//     std::chrono::duration<double> total = end - start;

//     int numKmers = std::accumulate(
//         contigs.begin(), contigs.end(), 0,
//         [](int sum, const std::list<kmer_pair>& contig) { return sum + contig.size(); });

//     if (run_type != "test") {
//         BUtil::print("Assembled in %lf total\n", total.count());
//     }

//     if (run_type == "verbose") {
//         printf("Rank %d reconstructed %d contigs with %d nodes from %d start nodes."
//                " (%lf read, %lf insert, %lf total)\n",
//                upcxx::rank_me(), contigs.size(), numKmers, start_nodes.size(), read.count(),
//                insert.count(), total.count());
//     }

//     if (run_type == "test") {
//         std::ofstream fout(test_prefix + "_" + std::to_string(upcxx::rank_me()) + ".dat");
//         for (const auto& contig : contigs) {
//             fout << extract_contig(contig) << std::endl;
//         }
//         fout.close();
//     }

//     upcxx::finalize();
//     return 0;
// }