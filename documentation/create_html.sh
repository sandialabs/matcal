#!/usr/bin/env bash
make clean
make html -j 8 

echo "launching a browser - escape if you can ..."
sleep 4
firefox build/html/index.html

