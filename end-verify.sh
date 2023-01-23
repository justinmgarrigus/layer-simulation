# end-verify.sh: Verifies that the execution process was successful. To run, do: 
#  $ ./end-verify.sh

# Checks how many of the generated error files are empty. 
num_empty=0
total_files=0
for file in $(find output -name "simErrors*" | sort); do
    echo -n "Checking file: ${file} ... "
    if [ -s ${file} ]; then 
        echo "not empty!"
    else 
        echo "empty"
        ((num_empty=num_empty+1))
    fi
    ((total_files=total_files+1)) 
done

echo -e "\n${num_empty}/${total_files} empty\n\n"

# Gets the duration of each model
for model_name in $(ls output); do 
    total_duration=0
    for file in $(find output/${model_name} -name "time*" | sort); do
        echo -n "Checking file: ${file} ... "
        duration=$(cat ${file} | grep -oP "Duration: \K\d+") 
        ((total_duration=total_duration+duration))
        duration_hours=$(bc <<< "scale=2; ${duration}/3600")
        echo "${duration} seconds (${duration_hours} hours)" 
    done
    duration_hours=$(bc <<< "scale=2; ${total_duration}/3600")
    echo -e "\n${model_name}: ${total_duration} seconds (${duration_hours} hours)\n\n"
done
