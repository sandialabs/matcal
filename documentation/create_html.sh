#!/usr/bin/env bash
cd ../../site_matcal/sandia/tests/unit
source ../../setup_env.sh
setup_env
cd ../../../../external_matcal/documentation
source ~/.bashrc || true
conda activate pyapprox-base || true

make clean
make html -j 8 

echo "launching a browser - escape if you can ..."
sleep 4
firefox build/html/index.html

