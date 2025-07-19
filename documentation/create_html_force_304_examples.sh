#!/usr/bin/env bash
make clean
make html -j 8 SPHINXOPTS="-D sphinx_gallery_conf.run_stale_examples=True -D sphinx_gallery_conf.filename_pattern='./advanced_examples/304L_viscoplastic_calibration/plot_'"

firefox build/html/index.html

