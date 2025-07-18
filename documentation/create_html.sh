#!/usr/bin/env bash
#export PYTHONPATH=$(pwd):/projects/granta/lib/python:/projects/seacas/linux_rhel7/current/lib:/projects/matcal/TPL/compadre:/projects/matcal/TPL/hierarchical-wavelet-decomposition:$PYTHONPATH
#export LD_LIBRARY_PATH=/projects/granta/lib/GRANTA_MIScriptingToolkit:$LD_LIBRARY_PATH
#
#export MATCAL_DIR=$(pwd)/../matcal/sandia
#
#
#alias matcal=$(pwd)/../matcal/sandia/matcal
#alias plot_matcal=$(pwd)/../matcal/sandia/plot_matcal
#
#module load dakota/6.19.0-aue
#module load anaconda3
#
#export PATH="/usr/netpub/texlive/current/bin/x86_64-linux/:$PATH"
#

make clean
make html -j 8 

firefox build/html/index.html

