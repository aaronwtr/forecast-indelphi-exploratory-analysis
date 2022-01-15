plotFreqIndels5D = function(data.indels1.no.cnt,
                            legend_lab = paste0("Number of gRNAs \n(",guidesPerSampleGroup(data.indels1.no.cnt)[, unique(N_guides)]," total) \n"),
                            cols = c(colorRampPalette(c("white", "#95cbee", "#0099dc", "#4ab04a", "#ffd73e"), bias = 3)(10))){
  # full topographic colors
  #cols = c(colorRampPalette(c("white", "#95cbee", "#0099dc", "#4ab04a", "#ffd73e"), bias = 2)(10), #"#c9e2f6"
  #         colorRampPalette(c("#eec73a", "#e29421", "#e29421", "#f05336","#ce472e"))(20))
  # only up to yellow
  #cols = c(colorRampPalette(c("white", "#95cbee", "#0099dc", "#4ab04a", "#ffd73e"), bias = 3)(10))
  breaks = c(50, 100, 250, 500, 750, 1000, 1500)
  if(max(data.indels1.no.cnt$class_of_dominant_indel_N) > 1500){
    breaks = c(breaks, 3000)
    if(max(data.indels1.no.cnt$class_of_dominant_indel_N) > 4000){
      breaks = c(breaks, seq(4000, max(data.indels1.no.cnt$class_of_dominant_indel_N), 2000))
    }
  }
  breaks = breaks[breaks <= max(data.indels1.no.cnt$class_of_dominant_indel_N)]
  labels = as.character(breaks)
  ggplot(data.indels1.no.cnt[!is.na(sample_class_of_dominant_indel) & !is.na(class_of_dominant_indel)],
         aes(sample_class_of_dominant_indel, class_of_dominant_indel)) +
    geom_raster(aes(fill = class_of_dominant_indel_N))+
    scale_fill_gradientn(colours=cols,
                         breaks=breaks, 
                         na.value=rgb(246, 246, 246, max=255),
                         labels=labels,
                         guide=guide_colourbar(ticks=T, nbin=50,
                                               barheight=.5, label=T))+
    ylab("Category of dominant mutation in K562") +
    xlab("Category of dominant mutation in other cells") +
    facet_wrap(~sample_group, nrow = 2) + theme_bw() +
    theme(legend.title = element_text(size = 12,face="bold"),
          legend.text = element_text(size = 10, angle = 90, hjust = 1, vjust = 0.5), 
          legend.position = "bottom",
          panel.grid = element_blank(),
          legend.key.width = unit(55, "pt"),
          plot.title = element_text(size=16,face="bold"),
          axis.title.y = element_text(hjust = 0),
          axis.title = element_text(size=12,face="bold"),
          axis.text = element_text(size = 10),
          axis.text.x = element_text(angle = 90, hjust = 1, vjust = 0.5),
          legend.margin = margin(0,0,10,0)) +
    labs(fill = legend_lab)
}