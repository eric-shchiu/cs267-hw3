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
#include <stdexcept> // Include stdexcept for runtime_error

#include "hash_map.hpp" // Should be included after system headers and upcxx
#include "kmer_t.hpp"
#include "read_kmers.hpp"
#include "butil.hpp"

int main(int argc, char** argv) {
    upcxx::init();

    // Declare hashmap instance here in the main scope for all ranks
    HashMap hashmap; // Default constructor

    if (argc < 2) {
        if (upcxx::rank_me() == 0) { // Print usage only once
                BUtil::print("usage: srun -N nodes -n ranks ./kmer_hash kmer_file [verbose|test [prefix]]\n");
        }
        upcxx::finalize();
        return 1; // Use return code 1 for error
    }

    std::string kmer_fname = std::string(argv[1]);
    std::string run_type = (argc >= 3) ? argv[2] : "";
    std::string test_prefix = (run_type == "test" && argc >= 4) ? argv[3] : "test";

    int ks = kmer_size(kmer_fname);
    if (ks != KMER_LEN) {
        // Use BUtil::print for rank 0 controlled printing or print from all ranks
            if (upcxx::rank_me() == 0) {
                BUtil::print("Error: %s contains %d-mers, while this binary is compiled for %d-mers. Modify packing.hpp and recompile.\n",
                            kmer_fname.c_str(), ks, KMER_LEN);
            }
            upcxx::finalize(); // Ensure finalize is called on error path
            return 1;
            // Or throw std::runtime_error after printing
            // throw std::runtime_error("Error: " + kmer_fname + " contains " + std::to_string(ks) +
            //                         "-mers, while this binary is compiled for " +
            //                         std::to_string(KMER_LEN) + "-mers. Modify packing.hpp and recompile.");
    }

    size_t n_kmers = line_count(kmer_fname);
    size_t hash_table_size = n_kmers * 2; // Load factor 0.5

    // Initialize the hashmap instance AFTER upcxx::init and AFTER checking args
    // This also sets HashMap::local_instance and creates the atomic domain
    hashmap.init(hash_table_size);
    // init includes barriers to ensure completion and visibility

    if (run_type == "verbose") {
        BUtil::print("Rank %d: Initialized hash table partition of size %lu for potentially %lu total kmers.\n",
                        upcxx::rank_me(), hashmap.my_size, n_kmers);
    }

    // Barrier after init (already in init, but belt-and-suspenders)
    upcxx::barrier();

    std::vector<kmer_pair> kmers = read_kmers(kmer_fname, upcxx::rank_n(), upcxx::rank_me());

    if (run_type == "verbose") {
        BUtil::print("Rank %d: Finished reading %lu kmers.\n", upcxx::rank_me(), kmers.size());
    }

    upcxx::barrier(); // Barrier after reading

    auto start = std::chrono::high_resolution_clock::now();

    // -------------------------Start: Fill the hash table and record the starting node-------------------------
    std::vector<kmer_pair> start_nodes;
    size_t insert_failures = 0; // Count failures
    for (auto& kmer : kmers) {
        bool success = hashmap.insert(kmer);
        if (!success) {
            // Don't throw immediately, maybe log or count? Table might just be full.
            // Let's count for now. A full table isn't necessarily a fatal error if handled.
            insert_failures++;
            // Optional: print a warning (might be too verbose)
            // BUtil::print("Warning: Rank %d failed to insert kmer %s\n", upcxx::rank_me(), kmer.kmer_str().c_str());
        } else {
                // Only add to start_nodes if insertion was successful
            if (kmer.backwardExt() == 'F') {
                start_nodes.push_back(kmer);
            }
        }
    }

    // Sum up insert failures across all ranks (optional but informative)
    size_t total_insert_failures = upcxx::reduce_all(insert_failures, upcxx::op_fast_add).wait();
    if (total_insert_failures > 0 && upcxx::rank_me() == 0) {
            BUtil::print("Warning: %lu kmer insertions failed across all ranks (hash table potentially full).\n", total_insert_failures);
    }
    // -------------------------End: Fill the hash table and record the starting node-------------------------

    auto end_insert = std::chrono::high_resolution_clock::now();
    upcxx::barrier(); // Barrier after insertion phase

    double insert_time = std::chrono::duration<double>(end_insert - start).count();
    if (run_type != "test") {
        BUtil::print("Rank %d: Finished inserting in %lf seconds (%lu failures)\n", upcxx::rank_me(), insert_time, insert_failures);
    }

    // Optional barrier
    // upcxx::barrier();
    auto start_read = std::chrono::high_resolution_clock::now();

    // -------------------------Start: Build the Chain table-------------------------
    std::list<std::list<kmer_pair>> contigs;
    size_t find_failures = 0; // Count find failures during contig building
    for (const auto& start_kmer : start_nodes) {
        std::list<kmer_pair> contig;
        contig.push_back(start_kmer); // Add the start kmer
        kmer_pair current_kmer = start_kmer; // Use a temporary to avoid modifying start_kmer

        // Loop while the last kmer in the contig has a forward extension
        while (current_kmer.forwardExt() != 'F') {
            pkmer_t next_key = current_kmer.next_kmer(); // Get the key for the next kmer
            kmer_pair next_val; // Variable to store the found kmer

            bool found = hashmap.find(next_key, next_val);

            if (!found) {
                    // *** This is where your original error happened ***
                    find_failures++;
                    // Print detailed error info
                if (run_type != "test") { // Avoid excessive printing during tests
                    kmer_pair tmp_debug; tmp_debug.kmer = next_key; // Temp just for printing string
                    BUtil::print("ERROR: Rank %d cannot find next_kmer during contig build:\n"
                                "  prev_kmer:  %s (%c%c)\n"
                                "  next_kmer:  %s (key)\n"
                                "  owner:      %d\n"
                                "  Contig len so far: %lu\n",
                                upcxx::rank_me(),
                                current_kmer.kmer_str().c_str(), current_kmer.backwardExt(), current_kmer.forwardExt(),
                                tmp_debug.kmer_str().c_str(), // Print the string of the kmer we *failed* to find
                                hashmap.get_owner(next_key),
                                contig.size());
                }

                // Decide how to handle: break the contig or throw error
                // Breaking the contig might be more robust if input data has errors/gaps
                // throw std::runtime_error("Error: k-mer not found during contig assembly.");
                break; // Stop extending this contig if the next kmer isn't found
            }

            // Found the next kmer, add it to the contig and update current
            contig.push_back(next_val);
            current_kmer = next_val; // Move to the next kmer for the next iteration
        }
        // Add the completed (or broken) contig to the list
        contigs.push_back(contig);
    }
    // -------------------------End: Build the Chain table-------------------------

    // Report find failures (optional)
    size_t total_find_failures = upcxx::reduce_all(find_failures, upcxx::op_fast_add).wait();
        if (total_find_failures > 0 && upcxx::rank_me() == 0) {
            BUtil::print("Warning: %lu kmer lookups failed during contig building across all ranks.\n", total_find_failures);
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