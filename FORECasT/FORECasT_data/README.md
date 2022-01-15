# SelfTarget
Capsule for processing and predicting CRISPR/Cas9-generated mutations.

The following are all run as part of this capsule:

1. An example of how raw reads from our experiment were processed into indel summary information, demonstrated on 6 constructs only. After running, the output for these steps is in indel_processing_example. 
2. An example of how raw reads from the previously published study by Van Overbeek et al 2016, as referenced in this paper were processed into indel summary information, for one gRNA only. After running, the output of these steps is found in endogenous_processing_example
4. The steps from summarized indel data (as generated above, but for all constructs), to paper figures. After running, the figures will be in the plots directory.
3. An example of the predictor run for one gRNA and compared vs the measured distribution. The output is also in plots. 
