readConsScores = function(gRNA_pos_38, gerp_file, cons_col_name = "conservation_score"){
  seqlevelsStyle(gRNA_pos_38) = "ensembl"
  
  cons_scores = import(gerp_file, selection = BigWigSelection(gRNA_pos_38), as = 'RleList')
  sum_score = numeric(length(gRNA_pos_38))
  for(chr in unique(seqnames(gRNA_pos_38))) {
    sum_score[as.logical(seqnames(gRNA_pos_38) == chr)] = sum(Views(cons_scores[[chr]], ranges(gRNA_pos_38[seqnames(gRNA_pos_38) == chr])))
  }
  mcols(gRNA_pos_38)[,cons_col_name] = sum_score
  gRNA_pos_38
  
  #cons_scores = import(gerp_file, selection = BigWigSelection(gRNA_pos_38), as = "GRanges")
  #cons_scores = as.data.table(cons_scores)
  #setnames(cons_scores, "score", cons_col_name)
  #gRNA_pos_38 = as.data.table(gRNA_pos_38)
  #merge(gRNA_pos_38, cons_scores, by = c("seqnames", "start", "end", "width", "strand"),
  #      all.y = TRUE, all.x = FALSE)
}