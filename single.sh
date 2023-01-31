#!/bin/bash

# single.sh: Runs the gemm executable on a batch of files. File batches are 
#   passed in either through: (1) standard input, or (2) command-line 
#   arguments. File batches must follow the format of "<weight input file> 
#   <x input file> <gemm output file> <result output file> <error output file> 
#   <time output file>", where each file is a string and files are separated by 
#   one space.
# Execute: 
#   ./single.sh
#   ./single.sh <model_name> <layer_index>


if [ ! -d "/bin" ]
then
    echo "No bin directory exists! Run the project with 'python3 run.py'"
    exit 1
fi

# Processes a batch of files. Five parameters must be passed in: 
#   1: weight input file 
#   2: x input file 
#   3: gemm output file
#   4: result output file 
#   5: error output file 
#   6: time output file
function process_batch() {
    echo -n "weights: $1, x: $2, results: $3, errors: $4, time: $5 ... "

    if [ ! -f $1 ] || [ ! -f $2 ] 
    then 
        echo -e "\nError: weight file and input(x) file must both exist!" 
        exit 1 
    fi 
    
    start_time=$(date +%s)
    ./build/gemm --w $1 --x $2 --o $3 > $4 2> $5
    end_time=$(date +%s)
    duration=$((end_time-start_time))
    
    mkdir -p $(dirname $6) 
    
    echo "Start: ${start_time}" > $6
    echo "End: ${end_time}" >> $6
    echo "Duration: ${duration}" >> $6

    echo "Done"
}

if [ $# -eq 2 ]
then 
    
    # User wants to only process one batch, where arguments are in the format
    # of "<model name> <layer number>" 
    
    range=$(ls bin/$1/gemm | wc -l) 
    if [ $2 -le 0 ] || [ $2 -gt ${range} ]
    then 
        echo "Error: index of $2 is out of range for model $1 (${range})" 
        exit 1 
    fi

    layer_name=$(ls bin/$1/gemm | sed -n $2p)
    layer_name="${layer_name%.*}" # removes file extension

    x_file="bin/$1/x/${layer_name}.bin"
    weight_file="bin/$1/weight/${layer_name}.bin"
    gemm_file="bin/$1/gemm/${layer_name}.bin" 
    result_file="output/$1/result/${layer_name}.txt"
    error_file="output/$1/error/${layer_name}.txt" 
    time_file="output/$1/time/${layer_name}.txt"
    
    process_batch      \
        ${weight_file} \
        ${x_file}      \
        ${gemm_file}   \
        ${result_file} \
        ${error_file}  \
        ${time_file}   

else 

    # Run in interactive mode, pulling from stdin. Lines inputted must follow
    # the format of: 
    #   <weight_file> <x_file> <gemm_file> <result_file> <error_file> <time_file> 

    while read -r line
    do 
        parts=($line)
        process_batch ${parts[@]} 
    done
        
fi 
