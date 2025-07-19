#!/usr/bin/env bash
make clean
make html -j 8 

firefox build/html/index.html

