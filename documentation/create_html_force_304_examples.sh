#!/usr/bin/env bash
#export PYTHONPATH=$(pwd):/projects/granta/lib/python:/projects/seacas/linux_rhel7/current/lib:$PYTHONPATH
#export LD_LIBRARY_PATH=/projects/granta/lib/GRANTA_MIScriptingToolkit:$LD_LIBRARY_PATH
#
#export MATCAL_DIR=$(pwd)/../matcal/sandia
#
#
#alias matcal=$(pwd)/../matcal/sandia/matcal
#alias plot_matcal=$(pwd)/../matcal/sandia/plot_matcal
#
#module load dakota/matcal_2022_09_28
#module load apps/anaconda3-2022.05
#
#export PATH="/usr/netpub/texlive/current/bin/x86_64-linux/:$PATH"

make clean
make html -j 8 SPHINXOPTS="-D sphinx_gallery_conf.run_stale_examples=True -D sphinx_gallery_conf.filename_pattern='./advanced_examples/304L_viscoplastic_calibration/plot_'"

firefox build/html/index.html

