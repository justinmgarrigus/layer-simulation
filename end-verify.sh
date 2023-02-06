#!/bin/bash 

# end-verify.sh: Verifies that the execution process was successful. To run, do: 
#  $ ./end-verify.sh
#  $ ./end-verify.sh <compare-bin-path> 
# Arguments: 
#  <compare-bin-path>: a path similar in structure to the standard bin/
#    directory, except that it contains different bin files collected via some 
#    separate process. Runs a routine that displays the distance between each 
#    bin/model-name/gemm/layer-name.bin file and 
#    compare-bin-path/model-name/gemm/layer-name.bin file 


# Checks how many of the generated error files are empty. 

num_empty=0
total_files=0
for model_name in $(ls output); do 
    for file in $(ls "output/${model_name}/error"); do
        echo -n "Checking file: ${file} ... "
        if [ -s ${file} ]; then 
            echo "not empty!"
        else 
            echo "empty"
            ((num_empty=num_empty+1))
        fi
        ((total_files=total_files+1)) 
    done
done 

echo -e "\n${num_empty}/${total_files} empty\n\n\n"


# Gets the duration of each model

for model_name in $(ls output); do 
    total_duration=0
    for file in $(find "output/${model_name}/time" -type f); do
        echo -n "Checking file: ${file} ... "
        duration=$(cat ${file} | grep -oP "Duration: \K\d+") 
        ((total_duration=total_duration+duration))
        duration_hours=$(bc <<< "scale=2; ${duration}/3600")
        echo "${duration} seconds (${duration_hours} hours)" 
    done
    duration_hours=$(bc <<< "scale=2; ${total_duration}/3600")
    echo -e "\n${model_name}: ${total_duration} seconds (${duration_hours} hours)\n\n"
done

echo


# Compares the similarity between the gemm files found in "output/name/gemm" and
# those found in "<compare-bin-path>/name/gemm"

if [ $# -eq 1 ]; then 
    for model_name in $(ls bin); do 
        for layer_name in $(ls "bin/${model_name}/gemm"); do
            echo -n "Distance between" \
                "bin/${model_name}/gemm/${layer_name} and" \
                "$1/${model_name}/gemm/${layer_name}: "
            
            distance=$(./build/gemm \
                --x "bin/${model_name}/gemm/${layer_name}" \
                --w "$1/${model_name}/gemm/${layer_name}" \
                --c \
            )
            
            echo ${distance} 
        done 
    done 
fi 