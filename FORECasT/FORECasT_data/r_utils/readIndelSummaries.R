readIndelSummaries = function(data.files, type = c("indels", "guides")[1], guide_list = NULL, inv = F){
  invert = function(x, inv) if(inv) return(!x) else return(x)
  all_data = lapply(data.files, function(data.file){
    ##########################################################################
    # read data
    ungzipped.file = gsub("\\.gz","",data.file)
    if(isTRUE(grepl("\\.gz",data.file))) 
      gunzip(data.file, ungzipped.file, remove = F, overwrite = T)
    data = fread(ungzipped.file, stringsAsFactors = F)
    if(isTRUE(grepl("\\.gz",data.file))) unlink(ungzipped.file)
    
    ##########################################################################
    # change column names to data.table-compatible format
    setnames(data, colnames(data),
             gsub(" ","_",colnames(data)))
    # filter data by list of guides
    if(!is.null(guide_list)) {
      ind = invert(x = data$Oligo_Id %in% guide_list, inv = inv)
      message(paste0(data.file, ": ", uniqueN(data$Oligo_Id[ind]), " gRNAs fit selection"))
      data = data[ind]
    }
    # remove collumns that are never used
    if(mean(c("Left", "Right", "Central") %in% colnames(data)) != 0)
      data[,c("Left", "Right", "Central") := NULL]
    if(mean(c("Altered_Sequence") %in% colnames(data)) != 0)
      data[,c("Altered_Sequence") := NULL]
    
    ##########################################################################
    # calculate sample names if not already present
    if(!"sample_ID" %in% colnames(data))
      data[,sample_ID := gsub("\\.txt","",basename(ungzipped.file))]
    if(!"sample_group" %in% colnames(data)){
      # Set sample groups
      data[, sample_group := gsub("^ST.+201(8_|7_)","",sample_ID)]
      data[, sample_group := gsub("_[^[:punct:]]+$","",sample_group)]
      data[, sample_group := gsub("_[^[:punct:]]+$","",sample_group)]
    }
    ##########################################################################
    if(type == "indels"){
      # Microhomology
      if(!"Microhomology_Group" %in% colnames(data)){
        # calculate microhomology size
        data[, Microhomology_Size := nchar(Microhomology_Sequence)]
        # remove collumns that are never used
        data[,c("Microhomology_Sequence") := NULL]
      }
      
      ##########################################################################
      # set indel class and a few (readable) indel classes
      if(!"indel_class" %in% colnames(data)) 
        data[, indel_class := paste0(Type, Size)]
      if(!"few_indel_classes" %in% colnames(data)) {
        # define indel classes
        data[Type == "I" & Size == 1, few_indel_classes := "I1"]
        data[Type == "I" & Size > 1, few_indel_classes := "I>1"]
        data[Type == "D" & Size == 1, few_indel_classes := "D1"]
        data[Type == "D" & Size == 2, few_indel_classes := "D2"]
        data[Type == "D" & Size > 2 & Microhomology_Size >= 2,
             few_indel_classes := "D>2, MH"]
        data[Type == "D" & Size > 2 & Microhomology_Size < 2,
             few_indel_classes := "D>2, no MH"]
        data[grepl("I", Most_Common_Indel) & grepl("D", Most_Common_Indel),
             few_indel_classes := "I+D"]
        # complex indels: D then I, I then D, D alone, I alone
        data[grepl("D", Most_Common_Indel), complex_indel_classes := "D alone"]
        data[grepl("I", Most_Common_Indel), complex_indel_classes := "I alone"]
        data[grepl("I", Most_Common_Indel) & grepl("D", Most_Common_Indel),
             complex_indel_classes := "Multiple events"]
      }
      
      ##########################################################################
      # normalised and cumulative reads
      if(!"Norm_MCI_Reads" %in% colnames(data)) # normalise by Total_reads (mutated reads)
        data[, Norm_MCI_Reads := MCI_Reads/Total_reads]
      if(!"Cum_Norm_MCI_Reads" %in% colnames(data)) {
        # order by read count
        setorder(data, sample_ID, Oligo_Id, -Norm_MCI_Reads)
        # Calculate cumulative read count
        data[, Cum_Norm_MCI_Reads := cumsum(Norm_MCI_Reads), by = .(sample_ID, Oligo_Id)]
      }
      if(!"reads_per_indel_class" %in% colnames(data)) {
        # sum up reads for indels of the same class within samples
        data[, reads_per_indel_class := sum(Norm_MCI_Reads), 
             by = c("Oligo_Id", "sample_ID", "few_indel_classes")]
        data[, reads_per_complex_indel_class := sum(Norm_MCI_Reads), 
             by = c("Oligo_Id", "sample_ID", "complex_indel_classes")]
      }
    }
    ##########################################################################
    data
  })
  Reduce(rbind, all_data)
}