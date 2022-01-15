caclMostFreqIndel5D = function(data_repl, ref_sample_group = "K562",
                               dominant_indel_freq = 0.2, normalise_guides = F,
                               class_of_dominant_indel_levels = c(few_indel_classes_levels, "none"),
                               few_indel_classes_levels = c("I1", "I>1", "D>=4, MH", "D<4, MH", "D>=4, no MH", "D<4, no MH"), 
                               sample_group_levels = c("CHO", "Mouse ESC", "Human iPSC", "RPE1", "HAP1", "eCAS9", "TREX2", "2A_TREX2")){
  # FIRST - find dominant indel classes in K562 controls and then merge this guide attribute to the data
  K562_control.data = classOfDominantIndel(data_repl, sel_sample_group = ref_sample_group, new_col = "class_of_dominant_indel", thresh = dominant_indel_freq, indel_class_lab = "few_indel_classes")
  K562_control.data = unique(K562_control.data[,.(Oligo_Id, class_of_dominant_indel)])
  
  # SECOND - find dominant indel classes in all other samples
  data.indels = lapply(data_repl[,unique(sample_group)], function(sample){
    ## All samples: label guides by class of dominant indel
    data.indels_temp = classOfDominantIndel(data_repl, sel_sample_group = sample, new_col = "sample_class_of_dominant_indel", thresh = dominant_indel_freq, indel_class_lab = "few_indel_classes")
    unique(data.indels_temp[,.(Oligo_Id, sample_class_of_dominant_indel, sample_group, Most_Common_Indel)])
  })
  data.indels = rbindlist(data.indels)
  
  # merge
  data.indels = merge(data.indels,
                      K562_control.data,
                      by = "Oligo_Id", all.x = T, all.y = F)
  
  # set order of indel classes
  data.indels[, class_of_dominant_indel := factor(class_of_dominant_indel,
                                                  levels = class_of_dominant_indel_levels)]
  data.indels[, sample_class_of_dominant_indel := factor(sample_class_of_dominant_indel,
                                                         levels = class_of_dominant_indel_levels)]
  
  # for each indel class in control K562 count how many guides are of each class in other samples groups
  data.indels1 = unique(data.indels[,.(Oligo_Id, class_of_dominant_indel, sample_class_of_dominant_indel, sample_group)])
  
  data.indels1[, class_of_dominant_indel_N := as.numeric(uniqueN(Oligo_Id)),
               by = .(sample_group, class_of_dominant_indel, sample_class_of_dominant_indel)]
  
  if(normalise_guides){
    # normalise by the number of guides in K562
    data.indels1[, class_of_dominant_indel_N := class_of_dominant_indel_N / uniqueN(Oligo_Id),
                 by = .(class_of_dominant_indel)]
  }
  
  # remove controls from plot
  data.indels1.no.cnt = data.indels1[!sample_group %in% c(ref_sample_group)]
  # set order of samples
  data.indels1.no.cnt[, sample_group := factor(sample_group,
                                               levels = sample_group_levels)]
  data.indels1.no.cnt
  
}