addIndelRepl = function(data){
  # normalise by Total_reads (mutated reads) 
  data = data[, sum_MCI_Reads := sum(MCI_Reads),
              by = .(Oligo_Id, Most_Common_Indel, sample_group)]
  
  data = data[, sum_Total_reads := {
    temp = unique(data.table(sample_ID = sample_ID,
                             Total_reads = Total_reads))
    sum(temp$Total_reads)
  }, by = .(Oligo_Id, sample_group)]
  
  data[, Norm_MCI_Reads := sum_MCI_Reads/sum_Total_reads]
  
  # remove sample-specific variables
  data[, c("sample_ID", "N_sample_ID", "Total_reads", "MCI_Reads",
           "Cum_Norm_MCI_Reads", "reads_per_indel_class",
           "reads_per_complex_indel_class") := NULL]
  data = unique(data)
  # order by read count
  setorder(data, sample_group, Oligo_Id, -Norm_MCI_Reads)
  # Calculate cumulative read count
  data[, Cum_Norm_MCI_Reads := cumsum(Norm_MCI_Reads),
       by = .(sample_group, Oligo_Id)]
  data[, mean_Norm_MCI_Reads := Norm_MCI_Reads]
  
  data
}