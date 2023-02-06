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


# This project may not be run from the base project directory; in that case, 
# we want to run the actual executable script from the same directoy.

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
#   7: base project directory (where .git is located) 
function process_batch() {
    echo "Processing batch:" 
    echo "  weights: $1"
    echo "  x: $2" 
    echo "  gemm: $3" 
    echo "  result: $4" 
    echo "  error: $5" 
    echo "  time: $6" 

    if [ ! -f $1 ] || [ ! -f $2 ] 
    then 
        echo -e "\nError: weight file and input(x) file must both exist!" 
        exit 1 
    fi 
    
    mkdir -p $(dirname $3) 
    mkdir -p $(dirname $4) 
    mkdir -p $(dirname $5) 
    mkdir -p $(dirname $6) 
    
    start_time=$(date +%s)
    $7/build/gemm --w $1 --x $2 --o $3 > $4 2> $5
    end_time=$(date +%s)
    duration=$((end_time-start_time))
    
    echo "Start: ${start_time}" > $6
    echo "End: ${end_time}" >> $6
    echo "Duration: ${duration}" >> $6

    echo "Done"
}

if [ $# -eq 2 ]
then 
    
    # User wants to only process one batch, where arguments are in the format
    # of "<model name> <layer number>" 
    
    project_dir=$(pwd)
    while [ ! -d $project_dir/.git ]; do 
        project_dir=$(dirname $project_dir)
        if [ -z $project_dir ]; then 
            echo "Error: single.sh must be run from the same directory as or" \
                 "from a subdirectory of the .git project". 
            exit 1 
        fi 
    done

    range=$(ls $project_dir/bin/$1/gemm | wc -l) 
    if [ $2 -le 0 ] || [ $2 -gt ${range} ]
    then 
        echo "Error: index of $2 is out of range for model $1 (${range})" 
        exit 1 
    fi

    layer_name=$(ls $project_dir/bin/$1/gemm | sed -n $2p)
    layer_name="${layer_name%.*}" # removes file extension

    x_file="$project_dir/bin/$1/x/${layer_name}.bin"
    weight_file="$project_dir/bin/$1/weight/${layer_name}.bin"
    gemm_file="$project_dir/bin/$1/gemm/${layer_name}.bin" 
    result_file="$project_dir/output/$1/result/${layer_name}.txt"
    error_file="$project_dir/output/$1/error/${layer_name}.txt" 
    time_file="$project_dir/output/$1/time/${layer_name}.txt"
    
    process_batch      \
        ${weight_file} \
        ${x_file}      \
        ${gemm_file}   \
        ${result_file} \
        ${error_file}  \
        ${time_file}   \
        ${project_dir} 

else 

    # Run in interactive mode, pulling from stdin. Lines inputted must follow
    # the format of: 
    #   <weight_file> <x_file> <gemm_file> <result_file> <error_file> <time_file> 

    while read -r line
    do 
        parts=($line $(pwd))
        process_batch ${parts[@]} 
    done
        
fi 
