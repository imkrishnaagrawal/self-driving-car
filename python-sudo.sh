#!/bin/bash

var=`sudo netstat -nlp | grep 8000`
code=`echo "$var" | awk '{n=split($0,a," "); print a[n]}' | awk '{n=split($0,a,"/"); print a[1]}'`
if [ ${#code} -ge 1 ]; then 
echo "$code";
sudo kill -9 $code;
fi

sudo /home/krishna/.virtualenvs/dl4cv/bin/python3 "$@"
