#!/usr/bin/env bash
#source ~/.bashrc
#
#export PYTHONPATH=$(pwd):/projects/granta/lib/python:/projects/seacas/linux_rhel7/current/lib:$PYTHONPATH
#export LD_LIBRARY_PATH=/projects/granta/lib/GRANTA_MIScriptingToolkit:$LD_LIBRARY_PATH
#
#export MATCAL_DIR=$(pwd)/../matcal/sandia
#
#
#alias matcal=$(pwd)/../matcal/sandia/matcal
#alias plot_matcal=$(pwd)/../matcal/sandia/plot_matcal
#
#module load dakota/matcal_2022_09_13
#module load apps/anaconda3-2022.05
#
#export PATH="/usr/netpub/texlive/current/bin/x86_64-linux/:$PATH"

make clean
make html-noplot

firefox build/html/index.html

