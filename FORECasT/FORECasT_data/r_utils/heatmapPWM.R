heatmapPWM = function(PWM){
  cols <- c(colorRampPalette(c("white", "#95cbee", "#0099dc", "#4ab04a", "#ffd73e"), bias = 2)(10), #"#c9e2f6"
            colorRampPalette(c("#eec73a", "#e29421", "#e29421", "#f05336","#ce472e"))(20))
  ggplot(PWM,
         aes(position, nucleotide)) +
    geom_raster(aes(fill = frequency))+
    scale_fill_gradientn(colours=cols, limits=c(0, 1),
                         breaks=c(0, 0.1, 0.2, 0.3, 0.4, 0.5, 1), 
                         na.value=rgb(246, 246, 246, max=255),
                         labels=c("0", "0.01", "0.05", "0.1", "0.2", "0.3", "1"),
                         guide=guide_colourbar(ticks=T, nbin=50,
                                               barheight=.5, label=T))+
    ylab("Nucleotide") +
    xlab(paste0("position")) +
    facet_wrap(~guide_class, nrow = uniqueN(PWM$guide_class)) + theme_bw() +
    theme(legend.title = element_text(size = 12,face="bold"),
          legend.text = element_text(size = 10, angle = 90, hjust = 1, vjust = 0.5), 
          legend.position = "bottom",
          panel.grid = element_blank(),
          legend.key.width = unit(55, "pt"),
          plot.title = element_text(size=16,face="bold"),
          axis.title.y = element_text(hjust = 0.5),
          axis.title = element_text(size=12,face="bold"),
          axis.text = element_text(size = 10),
          axis.text.x = element_text(angle = 0, hjust = 0.5, vjust = 0.5),
          legend.margin = margin(0,0,4,0)) +
    scale_x_discrete(labels = unique(PWM$position),
                     breaks = unique(PWM$position))
}