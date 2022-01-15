##' Find most 
findMostFreqPWM = function(string, indel_summaries, 
                           indel_filter = "I1", indel_col = "few_indel_classes",
                           guide_class = c(indel_filter)){
  all = findPWM(string, indel_summaries$Oligo_Id, guide_class, total_fun = rowSums, as.prob = FALSE)
  
  most_freq = indel_summaries[, .(max_Norm_MCI_Reads = as.numeric(max(mean_Norm_MCI_Reads))),
                        by = .(Oligo_Id, sample_group)]
  selected_indel = unlist(most_freq[, indel_col, with = FALSE]) == indel_filter
  selected_indel = findPWM(string, most_freq[selected_indel, Oligo_Id],
                           guide_class, total_fun = rowSums, as.prob = FALSE)
  res = merge(all, selected_indel, by = c("nucleotide", "position", "guide_class"))
  res[, c("frequency_all", "frequency_selected_top") := .(frequency.x, frequency.y)]
  res[, c("frequency.x", "frequency.y") := NULL]
  res[, frequency := frequency_selected_top / frequency_all]
  res
}