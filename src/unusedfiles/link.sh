#!/bin/bash
FILES=../data/dzne/linkdata/

for f in $FILES
do
  echo "Processing $f
  python sup_link_main_inclweeken.py 200 128 "l1" 0.001 "$f"

done
wait
exit 0
