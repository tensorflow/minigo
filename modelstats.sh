#!/bin/bash

for i in `ls $1 | tail -n 5`;
  do
  echo $i
  find sgf19/$i/ -name "*.sgf" | wc -l;
  echo -en "B+\t\t"
  find sgf19/$i/ -name "*.sgf" -print0 | xargs -0 grep -m 1 "B+" | wc -l
  echo -en "W+\t\t"
  find sgf19/$i/ -name "*.sgf" -print0 | xargs -0 grep -m 1 "W+" | wc -l
  echo -en "B+Resign\t"
  find sgf19/$i/ -name "*.sgf" -print0 | xargs -0 grep -m 1 "B+R" | wc -l
  echo -en "W+Resign\t"
  find sgf19/$i/ -name "*.sgf" -print0 | xargs -0 grep -m 1 "W+R" | wc -l
  #echo "Stats:"
  #find sgf19/$i/ -name "*.sgf" -exec /bin/sh -c 'tr -cd \; < {} | wc -c' \; | ministat -n 
  echo -en "\n"
done;

