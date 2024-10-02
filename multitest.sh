#!/bin/bash

# This script runs multiple equivilent random tests and takes the average of each test
# The number of tests is specified by the user in the first argument
# The test to be run is specified by the user in the second argument
# Optionally the seed for the random number generator is specified by the user in the third argument

# Check if the number of arguments is correct
if [ "$#" -ne 2 ]; then
    echo "Illegal number of parameters"
    echo "Usage: ./multitest.sh <repitions> <test to run> <seed>"
    exit 1
fi

# Check if the first argument is a number
if ! [[ "$1" =~ ^[0-9]+$ ]]; then
    echo "First argument must be a number"
    echo "Usage: ./multitest.sh <repitions> <test to run> <seed>"
    exit 1
fi

# Check if the third argument exists and is a number
if [ $3 ]; then
    if ! [[ "$3" =~ ^[0-9]+$ ]]; then
        echo "Third argument must be a number"
        echo "Usage: ./multitest.sh <repitions> <test to run> <seed>"
        exit 1
    fi
fi


# Run the tests
for i in $(seq 1 $1); do
    echo "Running test $i"
    if [ $3 ]; then python workexp.py with $2 seed=$3; else
    python workexp.py with $2; fi
done

#find the lastest output folder, which will be the largest numbered folder in the runs directory
run=$(ls runs | sort -n | tail -n 1)
echo Lastest output folder is $run



#move the output files from the different outputs generated by these tests to the same folder
#append the number of the test to the output files
for i in $(seq $(($run)) -1 $(($run-$1+1))); do
    echo "Moving output files from test $i to test $run"
    for f in runs/$i/*.npy; do mv "$f" "${f%}-${i}.npy"; done
    if ! [ i = $run ]; then mv runs/$i/*.npy runs/$run; fi
done

#run the average script on the output files
python multitest_average.py runs/$run