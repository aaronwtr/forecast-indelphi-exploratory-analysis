# adding equation and R^2 from https://stackoverflow.com/questions/7549694/adding-regression-line-equation-and-r2-on-graph
lm_eqn <- function(df, formula = y ~ x, grna_col = "ID", cor = T, by = list()){
  df[,.(label = {
    if(isTRUE(cor)){
      cor_res = cor.test(x = eval(as.list(formula)[[3]]),
                         y = eval(as.list(formula)[[2]]),
                         method = c("pearson"))
      eq = substitute("Pearson R"~"="~r2*","~~italic(p)~"="~p.val*","~~N_gRNA*~" gRNA", 
                       list(r2 = format(as.numeric(cor_res$estimate[1]), digits = 2),
                            p.val = format(cor_res$p.value[1], digits = 2),
                            N_gRNA = uniqueN(get(grna_col))))
    } else {
      m <- lm(formula, .SD);
      eq <- substitute(italic(y) == a + b %.% italic(x)*","~~italic(r)^2~"="~r2*","~~N_gRNA*" gRNA", 
                       list(a = format(coef(m)[1], digits = 2), 
                            b = format(coef(m)[2], digits = 2), 
                            r2 = format(summary(m)$r.squared, digits = 2),
                            N_gRNA = uniqueN(get(grna_col))))
    }
    as.character(as.expression(eq));                 
  }), by = by]
}

plotEffByFactor = function(data, formula = X1 ~ In.Frame.Percentage, 
                           fact_col = ~ targets_domain_lab,
                           fact_row = ~ .,
                           binwidth = NULL, bins = 20,
                           cor_lab_pos_x = 50, cor_lab_pos_y = 3.3,
                           cols = c(colorRampPalette(c("white", "#95cbee", "#0099dc", "#4ab04a", "#ffd73e"), bias = 3)(10))) {
  if(as.character(as.list(fact_row)[[2]]) != ".") { 
    group = paste0("interaction(", as.character(as.list(fact_col)[[2]]), ", ",
                   as.character(as.list(fact_row)[[2]]))
    dt_by = c(as.character(as.list(fact_col)[[2]]),
              as.character(as.list(fact_row)[[2]]))
  } else {
    group = as.list(fact_col)[[2]]
    dt_by = as.character(as.list(fact_col)[[2]])
  }
  X1plot_dom = ggplot(data,
                      aes(x = eval(as.list(formula)[[3]]),
                          y = eval(as.list(formula)[[2]]),
                          group = eval(group))) +
    geom_bin2d(binwidth = binwidth, bins = bins) +
    geom_smooth(method = "lm", se=FALSE, color="black") +
    scale_fill_gradientn(colours=cols, #trans = "log10",
                         #breaks = c(10, 1000, 100000, 1000000),
                         #labels = c("10", "1,000", "100,000", "1,000,000"),
                         na.value=rgb(246, 246, 246, max=255),
                         guide=guide_colourbar(ticks=T, nbin=50,
                                               barheight=10, label=T)) +
    scale_color_gradientn(colours=cols, #trans = "log10",
                          #breaks = c(10, 1000, 100000, 1000000),
                          #labels = c("10", "1,000", "100,000", "1,000,000"),
                          na.value=rgb(246, 246, 246, max=255),
                          guide=guide_colourbar(ticks=T, nbin=50,
                                                barheight=10, label=T))+
    xlab(paste0("Percentage of in-frame mutations per gRNA")) + 
    ylab(paste0("JACKs gRNA efficiency score")) +
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
  if(as.character(as.list(fact_row)[[2]]) != ".") {
    X1plot_dom = X1plot_dom + facet_grid(eval(as.list(fact_row)[[2]]) ~ eval(as.list(fact_col)[[2]]))
  } else {
    X1plot_dom = X1plot_dom + facet_wrap(~ eval(as.list(fact_col)[[2]]), ncol = 2)
  }
  X1plot_dom = X1plot_dom +geom_text(x = cor_lab_pos_x, y = cor_lab_pos_y,
              mapping = aes(label = label, group = eval(group)),
              data = lm_eqn(data, formula, by = dt_by),
              inherit.aes = FALSE, parse = TRUE)
}

plotEff = function(data, formula = X1 ~ In.Frame.Percentage,
                   binwidth = NULL, bins = 20,
                   cor_lab_pos_x = 50, cor_lab_pos_y = 3.3,
                   cols = c(colorRampPalette(c("white", "#95cbee", "#0099dc", "#4ab04a", "#ffd73e"), bias = 3)(10))) {
  X1plot = ggplot(data,
                  aes(x = eval(as.list(formula)[[3]]),
                      y = eval(as.list(formula)[[2]]))) +
    geom_bin2d(binwidth = binwidth, bins = bins) +
    geom_smooth(method = "lm", se=FALSE, color="black") +
    scale_fill_gradientn(colours=cols, #trans = "log10",
                         #breaks = c(10, 1000, 100000, 1000000),
                         #labels = c("10", "1,000", "100,000", "1,000,000"),
                         na.value=rgb(246, 246, 246, max=255),
                         guide=guide_colourbar(ticks=T, nbin=50,
                                               barheight=10, label=T)) +
    scale_color_gradientn(colours=cols, #trans = "log10",
                          #breaks = c(10, 1000, 100000, 1000000),
                          #labels = c("10", "1,000", "100,000", "1,000,000"),
                          na.value=rgb(246, 246, 246, max=255),
                          guide=guide_colourbar(ticks=T, nbin=50,
                                                barheight=10, label=T))+
    xlab(paste0("Percentage of in-frame mutations per gRNA")) + 
    ylab(paste0("JACKs gRNA efficiency score (X1)")) +
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
    labs(fill = "") +
    geom_text(x = cor_lab_pos_x, y = cor_lab_pos_y,
              mapping = aes(label = label),
              data = lm_eqn(data, formula),
              inherit.aes = FALSE, parse = TRUE)
}