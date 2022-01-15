loadGRNApos_kosuke = function(data.dir = "./data/", genes_set = NULL) {
  kosuke = fread(paste0(data.dir,"kosuke_ess_jacks_frame_merge.txt"), stringsAsFactors = FALSE)
  setnames(kosuke, colnames(kosuke), gsub(" ",".",colnames(kosuke)))
  if(!is.null(genes_set)) kosuke = kosuke[Gene %in% genes_set]
  kosuke[, chromosome_name := gsub("^.+\\.|\\:.+$", "", ID)]
  kosuke[, chromosome_name := gsub("^.+_", "", chromosome_name)]
  kosuke[, gRNA_start := gsub("^[^:]+\\:|\\-[[:digit:]]{5,9}(-|+).+$", "", ID)]
  kosuke[, gRNA_end := gsub(".+[[:digit:]]{5,9}\\-|\\:(-|+).+$", "", ID)]
  kosuke[, strand := gsub("^[^:]+\\:[[:digit:]]{5,9}\\-[[:digit:]]{5,9}\\:|_.+$", "", ID)]
  # find cut site position and record as "start/end BP |cut site|"
  kosuke[strand == "+", start := as.integer(gRNA_end) - 6]
  kosuke[strand == "-", start := as.integer(gRNA_start) + 6]
  kosuke[, end := start]
  kosuke[, gRNA_strand := strand]
  kosuke[, strand := "*"]
  makeGRangesFromDataFrame(kosuke,
                           keep.extra.columns=TRUE,
                           ignore.strand=FALSE,
                           seqinfo=NULL,
                           seqnames.field=c("seqnames", "seqname",
                                            "chromosome", "chrom",
                                            "chr", "chromosome_name",
                                            "seqid"),
                           start.field="start",
                           end.field=c("end", "stop"),
                           strand.field="strand",
                           starts.in.df.are.0based=FALSE)
}

loadGRNApos_gecko2 = function(data.dir = "./data/", genes_set = NULL) {
  gecko2_inframe = unique(fread(paste0(data.dir, "Gecko2_contexts_frame_shifts_mci.txt"),
                                stringsAsFactors = FALSE))
  setnames(gecko2_inframe, colnames(gecko2_inframe), gsub(" ",".",colnames(gecko2_inframe)))
  gecko2_jacks = fread(paste0(data.dir, "gecko2_grna_JACKS_results.txt"),
                       stringsAsFactors = FALSE)
  gecko2_pos = fread(paste0(data.dir, "Achilles_v3.3.8_sgRNA_mappings.txt"),
                     stringsAsFactors = FALSE)
  setnames(gecko2_pos, colnames(gecko2_pos), c("Guide", "chromosome_name","gRNA_start", "strand", "PAM"))
  
  gecko2 = merge(gecko2_inframe, gecko2_jacks,
                 by.x = "Guide.Sequence", by.y = "sgrna", all = F)
  gecko2[, ID := paste0(Guide.Sequence, "_", Gene)]
  gecko2 = merge(gecko2, gecko2_pos,
                 by.x = "ID", by.y = "Guide", all = F)
  
  if(!is.null(genes_set)) gecko2 = gecko2[Gene %in% genes_set]
  gecko2[, chromosome_name := gsub("^chr", "", chromosome_name)]
  gecko2[, gRNA_end := gRNA_start]
  # find cut site position and record as "start/end BP |cut site|"
  gecko2[strand == "+", start := as.integer(gRNA_start) - 2]
  gecko2[strand == "-", start := as.integer(gRNA_start) + 2]
  gecko2[, end := start]
  gecko2[, gRNA_strand := strand]
  gecko2[, strand := "*"]
  gecko2[, X2 := NA]
  makeGRangesFromDataFrame(gecko2,
                           keep.extra.columns=TRUE,
                           ignore.strand=FALSE,
                           seqinfo=NULL,
                           seqnames.field=c("seqnames", "seqname",
                                            "chromosome", "chrom",
                                            "chr", "chromosome_name",
                                            "seqid"),
                           start.field="start",
                           end.field=c("end", "stop"),
                           strand.field="strand",
                           starts.in.df.are.0based=FALSE)
}

loadGRNApos_avana = function(data.dir = "./data/", genes_set = NULL) {
  avana_inframe = unique(fread(paste0(data.dir, "Avana_sgrna_contexts_frame_shifts_mci.txt"), stringsAsFactors = FALSE))
  setnames(avana_inframe, colnames(avana_inframe), gsub(" ",".",colnames(avana_inframe)))
  avana_jacks = fread(paste0(data.dir, "avana_grna_JACKS_results.txt"), stringsAsFactors = FALSE)
  avana = merge(avana_inframe, avana_jacks,
                by.x = "Guide", by.y = "sgrna", all = T)
  setnames(avana, "Guide", "Guide.Sequence")
  avana = avana[!is.na(Locus)]
  if(!is.null(genes_set)) avana = avana[Gene %in% genes_set]
  avana[, ID := paste0(Guide.Sequence, Locus)] # this column is necessary for mapGRNA2protein()
  avana[, chromosome_name := gsub("^chr|_[[:digit:]]{5,10}_[-+]", "", Locus)]
  # find cut site position and record as "start BP |cut site|"
  avana[, start := gsub("^chr[[:alnum:]]{1,2}_|_[-+]", "", Locus)]
  avana[, end := start]
  avana[, gRNA_start := start]
  avana[, gRNA_end := end]
  avana[, gRNA_strand := gsub("^chr[[:alnum:]]{1,2}_[[:digit:]]{5,10}_", "", Locus)]
  avana[, strand := "*"]
  makeGRangesFromDataFrame(avana,
                           keep.extra.columns=TRUE,
                           ignore.strand=FALSE,
                           seqinfo=NULL,
                           seqnames.field=c("seqnames", "seqname",
                                            "chromosome", "chrom",
                                            "chr", "chromosome_name",
                                            "seqid"),
                           start.field="start",
                           end.field=c("end", "stop"),
                           strand.field="strand",
                           starts.in.df.are.0based=FALSE)
}