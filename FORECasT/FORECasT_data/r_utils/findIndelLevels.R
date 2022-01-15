findIndelLevels = function(data, gap = F){ # gap - gap betweeen D and I
  # pick the indel_class column
  data = unique(data[, .(indel_class)])
  indel_levels = unique(as.character(data[order( tstrsplit(indel_class, "[[:digit:]]{1,2}")[[1]],
                                                 as.integer(tstrsplit(indel_class, "[[:alpha:]]{1,1}")[2][[1]])), indel_class]))
  if(gap){
    return(c(indel_levels[grep("D", indel_levels)[length(grep("D", indel_levels)):1]],"",
             indel_levels[grep("I", indel_levels)]))
  }else{
    return(c(indel_levels[grep("D", indel_levels)[length(grep("D", indel_levels)):1]],
             indel_levels[grep("I", indel_levels)]))
  }
}