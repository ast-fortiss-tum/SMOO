#!/bin/bash
python -m black . --line-length 100 --preview

arr=("defaults" "src" "examples")
for elem in "${arr[@]}"
do
  filelist=$(mktemp)

  find "$elem" -type f -name "*.py" ! -path "*/_*/*" > "$filelist"

  if [ ! -s "$filelist" ]; then
    rm -f "$filelist"
    continue
  fi

  echo "Processing $elem..."
  darglint -s sphinx $(cat "$filelist")
  pyflakes $(cat "$filelist")
  isort --profile black $(cat "$filelist")

   # Run mypy except if folder is examples, they are not stable.
  if [[ "$elem" != "examples" ]]; then
    mypy --strict --follow-imports=skip @"$filelist"
  else
    echo "Skipping mypy check for $elem"
  fi

  rm -f "$filelist"
done