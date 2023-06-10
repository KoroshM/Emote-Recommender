#!/bin/bash

FILES="./data/*/*.txt"

for f in $FILES
do
  echo "Processing $f file..."

  # Parse time
  sed -re "s/\[([[:digit:]]+:[[:digit:]]+:[[:digit:]]+)\][[:space:]]/\1,/g" "$f" > "$f.csv"
  # Parse username
  sed -i -re "s/<(.+)>[[:space:]]/\1,/g" "$f.csv"
  # Add quotes around message
  sed -i -re "s/([^,]+,[^,]+,)(.*)(\n?)/\1\"\2\"\3/g" "$f.csv" 

done