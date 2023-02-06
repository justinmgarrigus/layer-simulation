#!/bin/bash

source ~/.bashrc 
source .env

echo "Enter: $1" 

while true; do 
    
    # Attempt to acquire the lock on ".commands"
    while true; do 
        
        # Link the file
        lock_name=".commands.$1.$RANDOM"
        link .commands $lock_name

        # Check if we're the only linker 
        if [ $(stat .commands --printf=%h) -eq 2 ]; then 
            
            # We're the only one with a link on the file at this exact moment.
            # Other process may link during the lifetime of our operation, but 
            # we will trust that they will do this same check
            break 

        fi 

        # Else, there was a collision. Remove our file and try again after 
        # some time
        unlink $lock_name
        duration=$(echo $RANDOM/32000 | bc -l) # Range [0.0, 1.0] 
        sleep $duration

    done 

    # We have acquired the lock: we can now process the file. Pop the first 
    # line from ".commands". 
    task=$(head -n 1 .commands) 
    regexp="^[\s]*[0-9]+[\s]*$"
    if [[ ! $task =~ $regexp ]]; then 
        exit 0 
    fi 
    tail -n +2 .commands > .commands.tmp 
    
    # Editing the file will change the link, so this needs to be our last
    # operation. Everyone will receive the updated content alongside the a
    # different stat link count. 
    mv .commands.tmp .commands
    unlink $lock_name
    
    # We no longer have the lock, so proceed with regular processing. 
    echo "Passing task to singularity: $task" 
    singularity exec --cleanenv \
        $MLNOC_SINGULARITY_PATH \
        ./worker.sh $task
        
    while true; do 

        lock_name="lines.$1.$RANDOM"
        link lines $lock_name 

        if [ $(stat lines --printf=%h) -eq 2 ]; then 

            break 

        fi 

        unlink $lock_name 
        duration=$(echo $RANDOM/32000 | bc -l) 
        sleep $duration 

    done 

    echo -e "$task\n$(cat lines)" > lines
    unlink $lock_name 

done 
