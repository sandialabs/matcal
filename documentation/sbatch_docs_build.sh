#!/bin/bash
#run using: sbatch -A fy220213 -N 1 -t 4:00:00 --partition batch,short sbatch_docs_build.sh
#run using: sbatch -A fy220213 -N 1 -t 24:00:00 --partition batch sbatch_docs_build.sh
#source ~/.bashrc
#module load aue/texlive/2023
./create_html.sh


