#!/usr/bin/env bash
make clean
make html-noplot

firefox build/html/index.html

