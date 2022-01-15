barplotPWM = function(PWM, group1, group2,
                      fill_colors = c("#FF7E0F", "#1F77B4", "#006400"),
                      ylab = "Proportion of gRNAs"){
  PWM = PWM[guide_class %in% c(group1, group2)]
  # overlapping histograms
  PWM.bar.plot = ggplot(aes(x = position, y = frequency,
                            fill = guide_class, color = guide_class), 
                        data = PWM) +
    theme_bw() +
    facet_grid(nucleotide ~ .)+
    geom_col(position = "identity", alpha= 0.5, color = "white") +
    ylab(ylab) + xlab("position") +
    theme(plot.title = element_text(size=14,face="bold"),
          axis.title = element_text(size=14,face="bold"),
          axis.title.y = element_text(size=12),
          axis.title.x = element_text(hjust = 1.2,size=0),
          axis.text = element_text(size = 10),
          axis.text.x = element_text(angle = 0, hjust = 0.5, vjust = 0.5),
          legend.position = c("bottom"),
          strip.text.y = element_text(angle = 0, size = 14),
          panel.grid.minor = element_line(size = 0),
          panel.grid.major.x = element_line(size = 0),
          plot.background = element_rect(fill="transparent", colour="transparent"),
          legend.background = element_rect(fill="transparent", colour="transparent")) +
    scale_x_discrete(labels = unique(PWM$position),
                     breaks = unique(PWM$position))+
    scale_color_manual(values=fill_colors)+
    scale_fill_manual(values=fill_colors)
  PWM.bar.plot
}