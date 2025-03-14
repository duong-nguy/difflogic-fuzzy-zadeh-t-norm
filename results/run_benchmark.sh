#!/bin/bash

# Constants
BATCH_SIZE=128
WARMUP_RUNS=10
TIMING_RUNS=90
NUM_BITS=64
COMPILER="gcc"
CFLAGS="-O2"

# Define dataset configurations
declare -A datasets=(
    ["monk1"]="17 2"
    ["monk2"]="17 2"
    ["monk3"]="17 2"
    ["adult"]="116 2"
    ["breast_cancer"]="51 2"
    ["mnist20x20"]="400 10"
    ["mnist"]="784 10"
    ["cifar-10-3-thresholds"]="9216 10"
    ["cifar-10-31-thresholds"]="95232 10"
)

# Library files to benchmark
declare -A lib_files=(
    ["monk1"]="00526000_64.so"
    ["monk2"]="00526001_64.so"
    ["monk3"]="00526002_64.so"
    ["adult"]="00526010_64.so"
    ["breast_cancer"]="00526020_64.so"
    ["mnist20x20"]="00526030_64.so"
    ["mnist"]="00526040_64.so"
    ["cifar-10-3-thresholds-1"]="00526050_64.so"
    ["cifar-10-3-thresholds-2"]="00526060_64.so"
    ["cifar-10-31-thresholds"]="00526070_64.so"
)

# Create benchmark C file
cat > benchmark.c << 'EOF'
#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include <stdbool.h>

// Configuration will be set via -D flags
extern void apply_logic_gate_net(bool const *inp, unsigned int *out, size_t len);

int main(void) {
    srand(time(NULL));  // Initialize random seed
    
    // Calculate batch size divisible by NUM_BITS
    size_t batch_size_div_bits = (BATCH_SIZE + NUM_BITS - 1) / NUM_BITS;
    size_t padded_batch_size = batch_size_div_bits * NUM_BITS;
    
    printf("========== BENCHMARK CONFIGURATION ==========\n");
    printf("Dataset: %s\n", DATASET_NAME);
    printf("Input dimensions: %d\n", NUM_INPUTS);
    printf("Number of classes: %d\n", NUM_CLASSES);
    printf("Bit size: %d\n", NUM_BITS);
    printf("Batch size: %d (padded to %ld)\n", BATCH_SIZE, padded_batch_size);
    printf("Warmup runs: %d\n", WARMUP_RUNS);
    printf("Timing runs: %d\n", TIMING_RUNS);
    printf("Library file: %s\n", LIB_FILE);
    printf("===========================================\n\n");
    
    // Allocate memory with the correct dimensions
    printf("Allocating memory...\n");
    bool *inp = malloc(padded_batch_size * NUM_INPUTS * sizeof(bool));
    unsigned int *out = malloc(padded_batch_size * NUM_CLASSES * sizeof(unsigned int));
    
    if (!inp || !out) {
        fprintf(stderr, "Memory allocation failed\n");
        return 1;
    }
    
    // Initialize inp with test data
    printf("Initializing test data...\n");
    for (size_t i = 0; i < padded_batch_size * NUM_INPUTS; ++i) {
        inp[i] = rand() % 2;  // Random 0 or 1
    }

    // Warmup runs
    if (WARMUP_RUNS > 0) {
        printf("Performing %d warmup runs...\n", WARMUP_RUNS);
        for (int i = 0; i < WARMUP_RUNS; i++) {
            apply_logic_gate_net(inp, out, batch_size_div_bits);
        }
    }
    
    // Timing runs
    printf("Performing %d timing runs...\n", TIMING_RUNS);
    struct timespec start, end;
    clock_gettime(CLOCK_MONOTONIC, &start);
    
    for (int i = 0; i < TIMING_RUNS; i++) {
        apply_logic_gate_net(inp, out, batch_size_div_bits);
    }
    
    clock_gettime(CLOCK_MONOTONIC, &end);
    
    // Calculate elapsed time in nanoseconds
    long long elapsed_ns = (end.tv_sec - start.tv_sec) * 1000000000LL + 
                         (end.tv_nsec - start.tv_nsec);
    
    // Calculate average time per run and per sample
    double avg_time_per_run_ns = (double)elapsed_ns / TIMING_RUNS;
    double avg_time_per_sample_ns = avg_time_per_run_ns / BATCH_SIZE;
    double samples_per_second = 1000000000.0 * BATCH_SIZE / avg_time_per_run_ns;
    
    printf("\n========== BENCHMARK RESULTS ==========\n");
    printf("Total time for %d runs: %lld ns\n", TIMING_RUNS, elapsed_ns);
    printf("Average time per run: %.2f ns\n", avg_time_per_run_ns);
    printf("Average time per sample: %.2f ns\n", avg_time_per_sample_ns);
    printf("Samples per second: %.2f\n", samples_per_second);
    
    // Write results to a file
    char filename[256];
    snprintf(filename, sizeof(filename), "benchmark_%s_%s.txt", DATASET_NAME, LIB_FILE);
    FILE *f = fopen(filename, "w");
    if (f) {
        fprintf(f, "Dataset: %s\n", DATASET_NAME);
        fprintf(f, "Input dimensions: %d\n", NUM_INPUTS);
        fprintf(f, "Number of classes: %d\n", NUM_CLASSES);
        fprintf(f, "Bit size: %d\n", NUM_BITS);
        fprintf(f, "Batch size: %d\n", BATCH_SIZE);
        fprintf(f, "Library file: %s\n", LIB_FILE);
        fprintf(f, "Total time (ns): %lld\n", elapsed_ns);
        fprintf(f, "Average time per run (ns): %.2f\n", avg_time_per_run_ns);
        fprintf(f, "Average time per sample (ns): %.2f\n", avg_time_per_sample_ns);
        fprintf(f, "Samples per second: %.2f\n", samples_per_second);
        fclose(f);
        printf("Results written to %s\n", filename);
    }
    
    // Clean up
    free(inp);
    free(out);
    printf("Benchmark completed successfully.\n");
    
    return 0;
}
EOF

# Function to run benchmark for a single dataset & library
run_benchmark() {
    local dataset=$1
    local lib_key=$2
    local lib_file=${lib_files[$lib_key]}
    
    # Parse dataset configuration
    read -r num_inputs num_classes <<< "${datasets[$dataset]}"
    
    echo "=========================================="
    echo "Benchmarking dataset: $dataset"
    echo "Library file: $lib_file"
    echo "Inputs: $num_inputs, Classes: $num_classes"
    
    # Compile the benchmark
    $COMPILER $CFLAGS benchmark.c -L. -Wl,-rpath,. -l:$lib_file -o benchmark_$dataset \
        -DDATASET_NAME=\"$dataset\" \
        -DNUM_INPUTS=$num_inputs \
        -DNUM_CLASSES=$num_classes \
        -DNUM_BITS=$NUM_BITS \
        -DBATCH_SIZE=$BATCH_SIZE \
        -DWARMUP_RUNS=$WARMUP_RUNS \
        -DTIMING_RUNS=$TIMING_RUNS \
        -DLIB_FILE=\"$lib_file\"
    
    if [ $? -ne 0 ]; then
        echo "Compilation failed for $dataset with $lib_file"
        return 1
    fi
    
    # Run the benchmark
    echo "Running benchmark..."
    ./benchmark_$dataset
    
    echo "Benchmark for $dataset with $lib_file completed"
    echo "=========================================="
    echo ""
}

# Run all benchmarks
run_all_benchmarks() {
    for dataset in "${!datasets[@]}"; do
        # Check if this dataset has a library file
        for lib_key in "${!lib_files[@]}"; do
            # Extract the base dataset name without variant
            base_dataset=$(echo "$lib_key" | cut -d'-' -f1,2,3)
            if [[ "$dataset" == "$base_dataset" || "$lib_key" == "$dataset"* ]]; then
                run_benchmark "$dataset" "$lib_key"
            fi
        done
    done
}

# Run a specific benchmark
run_specific_benchmark() {
    local dataset=$1
    local lib_file=$2
    
    # Find the lib_key for this lib_file
    local lib_key=""
    for key in "${!lib_files[@]}"; do
        if [[ "${lib_files[$key]}" == "$lib_file" ]]; then
            lib_key=$key
            break
        fi
    done
    
    if [[ -z "$lib_key" ]]; then
        echo "Library file $lib_file not found in configuration"
        return 1
    fi
    
    run_benchmark "$dataset" "$lib_key"
}

# Main execution
if [ $# -eq 0 ]; then
    echo "Running all benchmarks..."
    run_all_benchmarks
elif [ $# -eq 1 ]; then
    echo "Running benchmarks for dataset: $1"
    for lib_key in "${!lib_files[@]}"; do
        if [[ "$lib_key" == "$1"* ]]; then
            run_benchmark "$1" "$lib_key"
        fi
    done
elif [ $# -eq 2 ]; then
    echo "Running benchmark for dataset $1 with library $2"
    run_specific_benchmark "$1" "$2"
else
    echo "Usage: $0 [dataset] [library_file]"
    echo "       $0                 - Run all benchmarks"
    echo "       $0 dataset         - Run all benchmarks for dataset"
    echo "       $0 dataset lib.so  - Run specific benchmark"
    exit 1
fi

# Clean up temporary files but keep the results
rm -f benchmark.c
echo "Benchmark script completed."
