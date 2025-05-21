#!/bin/bash
python -m black . --line-length 100 --preview

arr=("defaults" "src" "examples")
for elem in "${arr[@]}"
do
  # Find all relevant Python files, excluding folders starting with "_"
  files=$(find "$elem" -type f -name "*.py" ! -path "*/_*/*")
  [ -z "$files" ] && continue # Skip if no matching files

  echo "Processing $elem..."

  darglint -s sphinx $files
  pyflakes $files
  isort --profile black $files
done