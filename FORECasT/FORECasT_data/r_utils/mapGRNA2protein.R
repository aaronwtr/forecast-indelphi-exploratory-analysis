mapGRNA2protein = function(gRNA_pos, edb,
                           targ_dom_name = "Targets domain",
                           not_targ_dom_name = "Does not target domain") {
  library(ensembldb)
  library(EnsDb.Hsapiens.v75)
  gnm_prt = genomeToProtein(gRNA_pos, edb)
  gnm_prt = lapply(seq_along(gnm_prt), function(i){
    new_mcols = as.data.table(mcols(gnm_prt[[i]]))
    new_mcols$protein_id = names(gnm_prt[[i]])
    new_mcols$start = start(gnm_prt[[i]])
    new_mcols$end = end(gnm_prt[[i]])
    new_mcols = new_mcols[grepl("ENSP", protein_id)]
    gRNA_mcols = as.data.table(mcols(gRNA_pos[i]))
    gRNA_mcols$seq_start = start(gRNA_pos[i])
    merge(new_mcols, gRNA_mcols, by = "seq_start", all.x = FALSE, all.y = TRUE)
  })
  gnm_prt = rbindlist(gnm_prt)
  # get domains in protein that map to gRNAs
  domains = select(edb, keys = gnm_prt$protein_id[!is.na(gnm_prt$protein_id)], keytype = "PROTEINID",
                   columns = c("UNIPROTID", "PROTEINID", "INTERPROACCESSION",
                               "PROTEINDOMAINID", "PROTDOMEND", "PROTDOMSTART", "PROTEINDOMAINSOURCE"))
  setnames(domains, "PROTEINID", "protein_id")
  gnm_prt = merge(gnm_prt, domains, by = "protein_id",
                  all.x = TRUE, all.y = FALSE, allow.cartesian=TRUE)
  # find which gRNAs target domains
  gnm_prt[start >= PROTDOMSTART & start <= PROTDOMEND, domain_targeted := TRUE]
  gnm_prt[is.na(domain_targeted), domain_targeted := FALSE]
  gnm_prt[, targets_domain := as.logical(max(domain_targeted)), by = .(ID)]
  gnm_prt[targets_domain == TRUE, targets_domain_lab := targ_dom_name]
  gnm_prt[targets_domain == FALSE, targets_domain_lab := not_targ_dom_name]
  # produce a simplified version of this table
  gnm_prt_simp = unique(gnm_prt[,.(ID, Guide.Sequence, Full.Context, Gene,
                                   In.Frame.Percentage, Most.Common.Indel, X1, X2,
                                   gRNA_start, gRNA_end, gRNA_strand, targets_domain, targets_domain_lab)])
  list(gnm_prt = gnm_prt, gnm_prt_simp = gnm_prt_simp, domains = domains)
}