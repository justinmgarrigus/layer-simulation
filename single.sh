#!/bin/bash

# single.sh: Runs the gemm executable on a batch of files. File batches are 
#   passed in either through: (1) standard input, or (2) through a file. File 
#   batches must follow the format of "<weight input file> <x input file> 
#   <result output file> <error output file> <time output file>", where each 
#   file is a string and files are separated by one space. 
# Execute: 
#   ./single.sh <batch_files_table.txt>


if [ ! -d "/bin" ]
then
	echo "No bin directory exists! Run the project with 'python3 run.py'"
	exit 1
fi

# Processes a batch of files. Five parameters must be passed in: 
#   1: weight input file 
#   2: x input file 
#   3: result output file 
#   4: error output file 
#   5: time output file
process_batch() {
    echo $#
    echo $0 
    echo $1
    exit 0  # TODO: Make sure that arguments work the way you think they will.

	parts=($line)

    weight_file=${parts[0]} 
    x_file=${parts[1]} 
    result_file=${parts[2]} 
    error_file=${parts[3]} 
    time_file=${parts[4]} 

	echo -n "weights: ${weight_file}, x: ${x_file} " 
    echo -n "results: ${result_file}, errors: ${error_file} "
    echo -n "time: ${time_file} ... "

	start_time=$(date +%s)

	./build/gemm --w ${weight_file} --x ${x_file} \
        > ${result_file} 2> ${error_file}
	
    end_time=$(date +%s)
	duration=$((end_time-start_time))

	echo "Start: ${start_time}" > ${time_file}
	echo "End: ${end_time}" >> ${time_file}
	echo "Duration: ${duration}" >> ${time_file}

	echo "Done"
}

if [ $# -eq 2 ]
then 
    
    # User wants to only process one batch, where arguments are in the format
    # of "<model name> <layer number>" 
    
    if [ ! -f "layer_index.txt" ] 
    then 
        echo -n "No layer index exists! A file named 'layer_index.txt' must " 
        echo "contain each parameter." 
        exit 1 
    fi

    layers=$(grep $1 layer_index.txt)
    line=$(echo ${layers} | sed "$2q;")

    echo "Arg: $1" 
    echo "Line: '${line}'"
    echo "Layers: '${layers}'" 
    echo "Match: $2q;d"
    exit 0 
fi 


if [ $# -eq 1 ]
    # User passed in a file containing
fi 

if [ $# -ge 1 -a -f "$1" ] && input="$1" || input="-"

for line in $(cat "$input") 
do

done
