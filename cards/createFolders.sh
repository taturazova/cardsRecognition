#!/bin/bash

# Define the range of numbers
numbers=$(seq -w 01 21)

# Define the prefixes
prefixes=("m")

# Loop through each prefix
for prefix in "${prefixes[@]}"; do
    # Loop through each number
    for number in $numbers; do
        # Create the folder name
        folderName="${prefix}${number}"
        # Create the folder
        mkdir -p "$folderName"
    done
done

# Move the files to their respective folders
for file in *.png; do
    # Extract the prefix and number from the file name
    prefix=${file:0:1}
    number=${file:1:2}
    
    # Construct the folder name
    folderName="${prefix}${number}"
    
    # Move the file to the appropriate folder
    if [ -d "$folderName" ]; then
        mv "$file" "$folderName/"
    fi
done