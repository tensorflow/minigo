# Copyright 2018 Google LLC
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#      http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

#!/bin/bash

set -e

DIR=$1
WORKDIR=`pwd`

cd $DIR

echo 'Total games'
ls -Rl * | grep .sgf | wc -l

echo 'Games with start at 5-5'
grep -l '^;B\[ee' **/*.sgf | wc -l

echo 'Games with start at 4-5'
R=`grep -l '^;B\[de' **/*.sgf | wc -l`
R=`expr $R + $(grep -l '^;B\[ed' **/*.sgf | wc -l)`
R=`expr $R + $(grep -l '^;B\[fe' **/*.sgf | wc -l)`
R=`expr $R + $(grep -l '^;B\[ef' **/*.sgf | wc -l)`
echo $R

echo 'Games with start at 4-4'
R=`grep -l '^;B\[dd' **/*.sgf | wc -l`
R=`expr $R + $(grep -l '^;B\[fd' **/*.sgf | wc -l)`
R=`expr $R + $(grep -l '^;B\[ff' **/*.sgf | wc -l)`
R=`expr $R + $(grep -l '^;B\[df' **/*.sgf | wc -l)`
echo $R

cd $WORKDIR
