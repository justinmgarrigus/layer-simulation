#!/bin/bash

# Script: "sub-run.sh"
# Usage: $ ./sub-run.sh [-c (recreate|ignore) | -e "command"]

# -c: "Create"
# Preconditions: "data/" should contain at least one image file.
# Description: For each image inside of "data/", we make a copy of the project
#   and place each copy inside of a new "sub-builds/" directory.

# -e "Execute" 
# Preconditions: "sub-builds/" contains at least one sub-build, created from 
#   the above step. 
# Description: Treats the argument (where "command" is a single string given 
#   in quotes) as a command, and executes it from the base directory of each 
#   sub-build in the "sub-builds/" directory. 
# Example: $ ./sub-run.sh -e "python3 run.py -alexnet"
#   This command runs the alexnet inference on each sub-build. Each sub-build
#   contains a single image in it, so a single inference is done in total. Each
#   sub-directory will contain the build results and output (e.g., if run with 
#   GPGPU-Sim). 

# Get execution mode via arguments
if [ "$1" == "-c" ]; then 
    if [ $# -eq 2 ]; then 
        if [[ "$2" -eq "recreate" || "$2" -eq "ignore" ]]; then 
            fileoption="$2"
        else 
            echo -n "Error: The parameter of the '-c' subcommand, if passed, "
            echo    "must be either 'recreate' or 'ignore'. Exiting."
            exit 1
        fi 
    else 
        fileoption=""
    fi
elif [ "$1" == "-e" ]; then 
    if [ $# -eq 2 ]; then 
        fileoption="$2"
    else 
        echo -n 'Error: the '-e' subcommand requires a string to be passed as '
        echo -n 'an argument that contains a command to be executed on each '
        echo    'sub-build. Exiting.' 
        exit 1
    fi 
else 
    echo "Error: either '-c' or '-e' must be specified. Exiting."
    exit 1
fi

# "Create" mode, to create the target subdirectories. 
if [ "$1" == "-c" ]; then

    # Files will be copied in the "sub-builds/" directory. 
    if [[ "$fileoption" == "" && -d "sub-builds/" ]]; then 
        echo -n "Warning: the 'sub-builds/' directory already exists. "
        echo    "What would you like to do?"
        select yn in "Re-create" "Ignore existing" "Exit"; do 
            case "$yn" in 
                "Re-create" ) 
                    echo -n "Choosing to re-create directories with conflicting "
                    echo -e "names.\n"
                    fileoption="recreate"
                    break ;; 
                "Ignore existing" ) 
                    echo -n "Choosing to ignore directories with conflicting " 
                    echo -e "names.\n"
                    fileoption="ignore"
                    break ;; 
                "Exit" )
                    echo "Exiting..."
                    exit 0 ;;
            esac
        done
    fi
    
    mkdir -p sub-builds
    
    # Find all images from the "data/" directory.
    images=$(find data -type f -name "*.jpg")
    for i in data/*.jpg ; do 
        # The name of the directory will be the name of the image (no 
        # extension, so "data/hello-world.jpg" becomes "hello-world"). 
        name=$(basename "$i" | sed 's/\(.*\)\..*/\1/')
        
        dir="sub-builds/$name/"
        if [ -d $dir ]; then 
            if [ $fileoption == "recreate" ]; then 
                echo -n "'$dir' already exists, recreating ... "
                rm -r "$dir"
                mkdir "$dir" 
            else
                echo "'$dir' already exists, skipping."
                continue
            fi
        else
            echo -n "Creating '$dir' ... " 
            mkdir "$dir"
        fi
        
        # Copy all the files/directories that don't begin with a dot (e.g., 
        # ".git") except for the "data/" and "sub-builds/" directories into 
        # this new directory. 
        find . -maxdepth 1 \
            ! \( -name ".*" -o -name "data" -o -name "sub-builds" \) \
            -exec cp -r "{}" "$dir" \;

        # Create the new "data" directory, and copy the single image into it. 
        mkdir "$dir/data"
        cp "$i" "$dir/data" 
    
        echo "done"
    done 

# "Execute" mode, to execute a command on each sub-directory. 
else

    if [ ! -d "sub-builds" ]; then 
        echo -n "Error: the 'sub-builds/' directory does not exist! Run the "
        echo    "command './sub-run.sh -c' first to generate it." 
        exit 1
    elif [ -z "$(ls -A sub-builds 2>/dev/null)" ]; then 
        echo -n "Error: the 'sub-builds/' directory does not contain any "
        echo -n "builds! Run the command './sub-run.sh -c' first (and make "
        echo -n "sure the 'data/' directory contains files!) to generate "
        echo    "builds."
        exit 1
    fi

    # For-each sub-directory, cd and run the command.
    run="cd \"{}\" && eval $fileoption" # "{}" is the sub-build to run on. 
    find sub-builds -maxdepth 1 -mindepth 1 -type d -print0 | \
        parallel --will-cite -0 "sh -c \"$run\""
    
fi 
