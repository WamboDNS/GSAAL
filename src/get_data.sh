#!/bin/bash

file_url="--url--"
destination_path="./experiments/datasets/"
zip_file_name="elki.zip"
extracted_folder_name="elki"

mkdir -p "$destination_path"
curl -o "${destination_path}${zip_file_name}" "$file_url"

if [ $? -eq 0 ]; then
    echo "Download completed successfully!"

    # Extract the contents of the zip file
    unzip -q "${destination_path}${zip_file_name}" -d "${destination_path}"

    # Check if the extraction was successful
    if [ $? -eq 0 ]; then
        echo "Extraction completed successfully!"
    else
        echo "Error: Unable to extract the zip file."
    fi

    rm "${destination_path}${zip_file_name}"

else
    echo "Error: Unable to download the file."
fi

cd experiments && python3 Download_datasets.py
