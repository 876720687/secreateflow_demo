#!/bin/bash


echo "demo2"
start=$(date +%s)
python demo2.py 10000
end=$(date +%s)
take=$(( end - start ))
echo Time taken to execute commands is ${take} seconds.


echo "demo3"
start=$(date +%s)
python demo3.py 10000
end=$(date +%s)
take=$(( end - start ))
echo Time taken to execute commands is ${take} seconds.


echo "demo4"
start=$(date +%s)
python demo4.py 10000
end=$(date +%s)
take=$(( end - start ))
echo Time taken to execute commands is ${take} seconds.
