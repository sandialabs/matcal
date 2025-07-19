#!/usr/bin/env bash
make clean
make html -j 8 SPHINXOPTS="-D sphinx_gallery_conf.run_stale_examples=True -D sphinx_gallery_conf.filename_pattern='./full_field_study_verification_examples/plot_'"

firefox build/html/index.html

