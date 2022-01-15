# Other functions
# function that selects only guides and repair results present in all samples
inAllSamples = function(indel_summaries, ...){ 
  indel_summaries = copy(indel_summaries)
  indel_summaries[ , N_sample_ID := uniqueN(sample_ID), by = ...]
  indel_summaries[N_sample_ID == uniqueN(indel_summaries$sample_ID)]
}
inAllSamplesGroups = function(indel_summaries, ...){ 
  indel_summaries = copy(indel_summaries)
  indel_summaries[ , N_sample_ID := uniqueN(sample_group), by = ...]
  indel_summaries[N_sample_ID == uniqueN(indel_summaries$sample_group)]
}
guidesPerSample = function(data) data[, .(N_guides = uniqueN(Oligo_Id)), by = sample_ID]
guidesPerSampleGroup = function(data) data[, .(N_guides = uniqueN(Oligo_Id)), by = sample_group]