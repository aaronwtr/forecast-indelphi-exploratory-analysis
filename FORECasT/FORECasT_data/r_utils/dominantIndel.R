# dominantIndel functions
chooseLab = function(indel_class_lab, fraq, thresh = 0.3){
  if(isTRUE(max(fraq) >= thresh)) unique(indel_class_lab[fraq == max(fraq)])[1] else "none"
}
classOfDominantIndel = function(data, sel_sample_group = "K562",
                                new_col = "class_of_dominant_indel", thresh = 0.20,
                                indel_class_lab = "few_indel_classes"){
  ## label guides by class of dominant indel
  data1 = unique(data[sample_group == sel_sample_group,
                      .(Oligo_Id, Most_Common_Indel, Norm_MCI_Reads,
                        indel_class_lab = get(indel_class_lab), sample_group, sample_ID)])
  ## label guides by class of dominant indel
  data1[, max_Norm_MCI_Reads := max(Norm_MCI_Reads),
        by = .(Oligo_Id, sample_ID)]
  data1 = data1[max_Norm_MCI_Reads == Norm_MCI_Reads]
  #data1[, sample_support := uniqueN(sample_ID),
  #                  by = .(Oligo_Id, Most_Common_Indel)]
  data1[, sample_support := sum(Norm_MCI_Reads >= thresh),
        by = .(Oligo_Id, Most_Common_Indel)]
  # when many max frequency indels keep the one with max support for that gRNA
  data1[, max_sample_support := max(sample_support),
        by = .(Oligo_Id)]
  data1 = data1[max_sample_support == sample_support]
  data1[, c(new_col) := "none"]
  data1[sample_support == uniqueN(sample_ID),
        c(new_col) := indel_class_lab]
  setnames(data1,"indel_class_lab", indel_class_lab)
  data1
}
dominantIndelClass = function(data, new_col = "dominant_indel_class", thresh = 0.5, indel_class_lab = "few_indel_classes"){
  # sum up reads for indels of the same class within samples
  data[, reads_per_indel_class := sum(mean_Norm_MCI_Reads), 
       by = c("Oligo_Id", "sample_group", indel_class_lab)]
  ## label guides by dominant indel class
  data[, c(new_col) := chooseLab(eval(parse(text = indel_class_lab)), reads_per_indel_class, thresh = thresh), 
       by = .(Oligo_Id)]
  data
}