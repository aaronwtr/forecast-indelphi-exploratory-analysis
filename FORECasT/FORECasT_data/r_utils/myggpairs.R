
myggpairs = function(matr, cols = NULL,
                     order_by_clust = T, order_manual = 1:ncol(matr),
                     title = "", corMethod = "pearson",
                     bins_2d = 20, low_2d = "#132B43", high_2d = "#56B1F7",
                     density_col = "royalblue1", density_fill = "royalblue1",
                     density_size = 1.2, density_vline_col = "black",
                     filename = "./pairs_plot", device = "svg",
                     width = 15, height = 15, units = "in",
                     line_01 = T, lim = NULL, save = T, theme = NULL, upper_text_col = "black"){
  
  d2_bin <- function(data, mapping, ..., bins = 20, low = "#132B43", high = "#56B1F7", line_01 = F, lim = NULL, theme_arg = theme) {
    g = ggplot(data = data, mapping = mapping) +
      geom_bin2d(..., aes(colour = ..count.., fill = ..count..)) +
      scale_fill_gradient(low = low, high = high)
    if(isTRUE(line_01)) g = g + geom_abline(slope = 1, intercept = 0, colour = "#191970")
    if(!is.null(lim)) g = g + xlim(lim[1], lim[2]) + ylim(lim[1], lim[2]) 
    if(!is.null(theme_arg)) g = g + theme_arg
    g
  }
  my_density <- function(data, mapping, lim = NULL, theme_arg = theme, ...) {
    #stop(as.character(mapping))
    xintercept = as.numeric(median(data[,gsub("\\`","",gsub("~","",as.character(mapping)))]))
    
    g = ggplot(data = data, mapping = mapping) +
      geom_density(..., col = density_col, fill = density_fill, size = density_size) +
      geom_vline(xintercept = xintercept, col = density_vline_col,
                 size = density_size)
    if(!is.null(lim)) g = g + xlim(lim[1], lim[2])
    if(!is.null(theme_arg)) g = g + theme_arg
    g
  }
  
  if(!is.null(cols)) {
    cols = colnames(matr) %in% cols
    matr = matr[,cols]
  }
  if(isTRUE(order_by_clust)){
    order = hclust(as.dist(1+cor(matr, method = corMethod)))$order  #kendall dist or pearson dist
    # order = hclust(dist(t(mat*exp_factor)))$order #euclidian dist
    mat_4_plot = as.data.frame(matr)[,order]
  } else {
    mat_4_plot = as.data.frame(matr)[,order_manual]
  }
  
  pairs_plot = GGally::ggpairs(data = mat_4_plot,
                               title = title,
                               lower = list(continuous = GGally::wrap(d2_bin,
                                                                      line_01 = line_01,
                                                                      lim = lim)),
                               diag = list(continuous = GGally::wrap(my_density,
                                                                     lim = lim)),
                               upper = list(continuous = "cor", corMethod = corMethod, color = upper_text_col)) #kendall dist or pearson dist
  
  # Correlation matrix plot
  p2 = GGally::ggcorr(mat_4_plot, method = c("pairwise", corMethod), label = TRUE, label_round = 2) #kendall dist or pearson dist
  # Get list of colors from the correlation matrix plot
  g2 = ggplotGrob(p2)
  colors <- g2$grobs[[6]]$children[[3]]$gp$fill
  # Change background color to tiles in the upper triangular matrix of plots 
  idx = 1
  p = ncol(mat_4_plot)
  for (k1 in 1:(p-1)) {
    for (k2 in (k1+1):p) {
      plt = GGally::getPlot(pairs_plot,k1,k2) 
      if(!is.null(theme)) plt = plt + theme
      plt = plt + theme(panel.background = element_rect(fill = colors[idx], color="white"),
                        panel.grid = element_line(color=colors[idx]),
                        panel.grid.major = element_line(color=colors[idx]),
                        panel.grid.minor = element_line(color=colors[idx]))
      
      pairs_plot <- GGally::putPlot(pairs_plot,plt,k1,k2)
      idx = idx+1
    }
  }
  
  if(isTRUE(save)){
  dir = gsub(basename(filename),"", filename)
  if(!dir.exists(dir)) dir.create(dir, recursive = T)
  suppressWarnings({
    ggsave(filename = filename,
           plot = pairs_plot, device = device,
           width = width, height = height, units = units)
  })
  }
  return(pairs_plot)
}

#set.seed(1)
#myggpairs(matrix(rnorm(1000*10), nrow = 1000, ncol = 10))
#myggpairs(matrix(rnorm(1000*10), nrow = 1000, ncol = 10),
#          order_by_clust = F, order_manual = paste0("V",1:10))
