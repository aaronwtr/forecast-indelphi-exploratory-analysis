plotSampleScatter = function(pairs_list, binwidth = NULL, bins = NULL,
                             wrap_ncol = 2,
                             plot_fun = geom_bin2d, plot_type = c("xy", "MA", "per_gRNA_cor")[1],
                             cor_lab_pos_x = NA, cor_lab_pos_y = NA,
                             no_split_by = F, # when plot type is per_gRNA_cor, do not split by indel class
                             indel_colors = NULL,
                             cols = c(colorRampPalette(c("white", "#95cbee", "#0099dc", "#4ab04a", "#ffd73e"), bias = 3)(10))){
  
  group = parse(text = paste0("interaction(type, ", pairs_list$split_by,")"))
  
  # if plot is MA calculate normalised difference (y) vs mean (x) for each samples being compared
  if(plot_type == "MA"){
    # calculate mean
    pairs_list$repl.pairs[, mean := mean(c(get(pairs_list$x_name), get(pairs_list$y_name))),
                          by = .(Oligo_Id, Most_Common_Indel, type)]
    print(pairs_list$repl.pairs[Oligo_Id == Oligo_Id[1] & Most_Common_Indel == Most_Common_Indel[1]])
    # calculate normalised difference (y)
    pairs_list$repl.pairs[, diff := (get(pairs_list$y_name) - get(pairs_list$x_name)) / mean]
    pairs_list$repl.pairs[, c(pairs_list$x_name, pairs_list$y_name) := .(mean, diff)]
    # change labels
    pairs_list$repl.labels
  }
  # if plot is per_gRNA_cor, calculate per gRNA correlation for each samples being compared and plot densities
  # also put mean R into pairs_list$repl.labels
  if(plot_type == "per_gRNA_cor"){
    suppressWarnings({
      if(no_split_by){
        group = parse(text = paste0("interaction(type)"))
        # calculate per gRNA correlation
        pairs_list$repl.pairs[, cor := cor(get(pairs_list$x_name), get(pairs_list$y_name),
                                           method = "pearson", use = "all.obs"),
                              by = .(Oligo_Id, type)]
        pairs_list$repl.pairs[, cor_by_class := median(cor, na.rm = T),
                              by = .(type)]
        pairs_list$repl.pairs[, cor_label := paste0("R: ", signif(cor_by_class, 2))]
        pairs_list$repl.pairs[, c(pairs_list$split_by) := "gRNA"]
      } else {
        # calculate per gRNA correlation
        pairs_list$repl.pairs[, cor := cor(get(pairs_list$x_name), get(pairs_list$y_name),
                                           method = "pearson", use = "all.obs"),
                              by = .(Oligo_Id, type, get(pairs_list$split_by))]
        pairs_list$repl.pairs[, cor_by_class := median(cor, na.rm = T),
                              by = .(get(pairs_list$split_by), type)]
        pairs_list$repl.pairs[, cor_label := paste0("R: ", signif(cor_by_class, 2))]
      }
    })
    pairs_list$repl.pairs = unique(pairs_list$repl.pairs[,.(Oligo_Id, type, cor, cor_label,
                                                            split_by = get(pairs_list$split_by))])
    setnames(pairs_list$repl.pairs, c("split_by"), c(pairs_list$split_by))
    # change labels
    pairs_list$repl.pairs[, type := gsub(", R: [[:digit:]]\\.[[:digit:]]+", "", type)]
    pairs_list$repl.pairs[, type := paste0(type, ", R: ", signif(median(cor, na.rm = T), 2)), by = .(type)]
    pairs_list$repl.labels = unique(pairs_list$repl.pairs[, .(x = 0.35, y = 0.8, type,
                                                              split_by = get(pairs_list$split_by), cor_label)])
    setnames(pairs_list$repl.labels, c("split_by"), c(pairs_list$split_by))
    all_freq.plot = ggplot(pairs_list$repl.pairs,
                           aes(x = cor,
                               group = eval(group),
                               color = get(pairs_list$split_by), fill = get(pairs_list$split_by))) +
      geom_histogram(binwidth = binwidth, bins = bins) +
      labs(fill = "", color = "")
    if(!no_split_by){
      all_freq.plot = all_freq.plot +
        scale_color_manual(values=indel_colors) +
        scale_fill_manual(values=indel_colors) 
    }
  } else {
    all_freq.plot = ggplot(pairs_list$repl.pairs,
                           aes(x = get(pairs_list$x_name),
                               y = get(pairs_list$y_name),
                               group = eval(group))) +
      plot_fun(binwidth = binwidth, bins = bins) +
      scale_x_continuous(labels = scales::percent) +
      scale_y_continuous(labels = scales::percent) +
      scale_fill_gradientn(colours=cols, trans = "log10",
                           breaks = c(10, 1000, 100000, 1000000),
                           labels = c("10", "1,000", "100,000", "1,000,000"),
                           na.value=rgb(246, 246, 246, max=255),
                           guide=guide_colourbar(ticks=T, nbin=50,
                                                 barheight=20, label=T)) +
      scale_color_gradientn(colours=cols, trans = "log10",
                            breaks = c(10, 1000, 100000, 1000000),
                            labels = c("10", "1,000", "100,000", "1,000,000"),
                            na.value=rgb(246, 246, 246, max=255),
                            guide=guide_colourbar(ticks=T, nbin=50,
                                                  barheight=20, label=T))
  }
  
  all_freq.plot = all_freq.plot +
    xlab(paste0("Percentage of mutations per gRNA (",pairs_list$x_name,")")) + 
    ylab(paste0("Percentage of mutations per gRNA (",pairs_list$y_name,")")) +
    theme_bw() +
    theme(legend.title = element_text(size = 12,face="bold"),
          legend.text = element_text(size = 12), 
          legend.position = "right",
          plot.title = element_text(size=16,face="bold"),
          axis.title=element_text(size=12,face="bold"),
          axis.text = element_text(size=11),
          axis.text.x = element_text(angle=-45, hjust = 0, vjust = 0.7),
          panel.grid.minor = element_line(size = 0),
          panel.grid.major = element_line(size = 0.2),
          strip.text.y = element_text(size=11, angle = 0),
          strip.text.x = element_text(size=11)) +
    labs(fill = "")
  if(uniqueN(pairs_list$repl.pairs$type) == 1){
    all_freq.plot = all_freq.plot + 
      facet_wrap(eval(parse(text = paste0("~", pairs_list$split_by))), ncol = wrap_ncol) +
      ggtitle(unique(pairs_list$repl.pairs$type))
  } else if(uniqueN(pairs_list$repl.pairs[, get(pairs_list$split_by)]) == 1) {
    all_freq.plot = all_freq.plot + 
      facet_wrap(~ type, ncol = wrap_ncol)
  } else {
    all_freq.plot = all_freq.plot + 
      facet_grid(eval(parse(text = paste0(pairs_list$split_by, "~ type"))))
  }
  # change correlation label position if provided
  if(!is.na(cor_lab_pos_x)) pairs_list$repl.labels$x = cor_lab_pos_x
  if(!is.na(cor_lab_pos_y)) pairs_list$repl.labels$y = cor_lab_pos_y
  all_freq.plot = all_freq.plot +
    geom_text(mapping = aes(x = x, y = y, 
                            label = cor_label,
                            group = eval(group)),
              data = pairs_list$repl.labels, size = 5, inherit.aes = F, nudge_x = 0.02)
  all_freq.plot
}
