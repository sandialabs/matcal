#!/usr/bin/env bash

#module load matcal/current
#export PATH="/usr/netpub/texlive/current/bin/x86_64-linux/:$PATH"

make clean
make latex

# need to ensure correct pathing when compiling latex.
# export PATH="/usr/netpub/texlive/current/bin/x86_64-linux/:$PATH"
# current version of sphinx uses asmath.sty, need to manually change it to amsmath.sty

