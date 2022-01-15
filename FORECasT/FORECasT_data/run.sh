#!/bin/bash
##set -ex 

#Compile the indelmap and indelgen C++ code:
# custom code used to align reads to determine indels,
# and generate indels (called from python code below).
cd /code/indel_analysis/indelmap
mkdir build
cd build
cmake .. -DINDELMAP_OUTPUT_DIR=/usr/local/bin/
make
make install
./indelmaptest

#Compile the python libraries (utility functions used below)
cd /code/selftarget_pyutils
pip install .
cd /code/indel_prediction
pip install .

#Run the example that converts raw reads to summarized indels
cd /code/indel_analysis/compute_indels
python run_example.py                   ##Example on small set of 6 oligos of 
                                        ##extraction of indel summaries

#Endgenous vs Synthetic endogenous_comparisons
cd /code/indel_analysis/endogenous_comparisons
python run_example.py                   ##Single example of extraction of indels 
                                        ##from data from van Overbeek et al 2016.
python compare_overbeek_profiles.py     ##Generation of plots in paper Fig2 from 
                                        ##pre-generated indels.

#Generation of other plots detailing indel characteristics
cd /code/indel_analysis
python plot_all.py

cd /code
Rscript process_Rmd.R

#Example of indel prediction
cd /code/indel_prediction/model_testing
python compute_predicted_old_new_kl.py 

