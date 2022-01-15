findPWM = function(string, oligos, guide_class, total_fun = rowMeans, as.prob = TRUE){
  PWM = consensusMatrix(string[oligos], as.prob = as.prob)[DNA_BASES,]
  total = total_fun(PWM)
  PWM = cbind(PWM, total)
  PWM = as.data.table(PWM, keep.rownames = "nucleotide")
  PWM = melt.data.table(PWM, id.vars = "nucleotide",
                        value.name = "frequency", variable.name = "position")
  PWM[, position := gsub("V","", position)]
  PWM[, guide_class := guide_class]
  PWM
}

diffPWM = function(PWM1, PWM2, mode = c("abs_diff", "substract")[1]) {
  PWM = merge(PWM1, PWM2, by = c("nucleotide","position"))
  PWM[, frequency := frequency.x - frequency.y]
  PWM[, guide_class := paste0(guide_class.x," - ",guide_class.y)]
  if(isTRUE(mode == "abs_diff")) {
    PWM[, frequency := abs(frequency)]
    PWM[, guide_class := paste0("abs(",guide_class, ")")]
  }
  PWM[,.(nucleotide, position, frequency, guide_class)]
}

findOligoPWM = function(string, oligos, guide_class, width = 2, as.prob = TRUE){
  dinucs = lapply(1:(width(string)[1] - width), function(i){
    dinuc = nucleotideFrequencyAt(string[oligos], i:(i-1+width), as.prob = as.prob)
    data.table(nt1 = rep(rownames(dinuc), each = 4),
               nt2 = rep(colnames(dinuc), times = 4),
               position = i,
               frequency = c(dinuc[DNA_BASES[1],], dinuc[DNA_BASES[2],], dinuc[DNA_BASES[3],], dinuc[DNA_BASES[4],]))
  })
  dinucs = Reduce(rbind, dinucs)
  dinucs[, guide_class := guide_class]
  dinucs[, nucleotide := paste0(nt1, nt2)]
  dinucs[, c("nt1", "nt2") := NULL]
  total = unique(dinucs[,data.table(position = "total",
             frequency = mean(frequency),
             guide_class = guide_class), by = nucleotide])
  dinucs = rbind(dinucs, total)
  dinucs
}
