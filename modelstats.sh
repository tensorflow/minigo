#!/bin/bash

for i in `ls sgf19 | tail`;
  do
  echo $i
  find sgf19/$i/ -name "*.sgf" | wc -l;
  echo -en "B+\t\t"
  find sgf19/$i/ -name "*.sgf" -print0 | xargs -0 grep "B+" | wc -l
  echo -en "W+\t\t"
  find sgf19/$i/ -name "*.sgf" -print0 | xargs -0 grep "W+" | wc -l
  echo -en "B+Resign\t"
  find sgf19/$i/ -name "*.sgf" -print0 | xargs -0 grep "B+R" | wc -l
  echo -en "W+Resign\t"
  find sgf19/$i/ -name "*.sgf" -print0 | xargs -0 grep "W+R" | wc -l
  #echo "Stats:"
  #find sgf19/$i/ -name "*.sgf" -exec /bin/sh -c 'tr -cd \; < {} | wc -c' \; | ministat -n 
  echo -en "\n"
done;

