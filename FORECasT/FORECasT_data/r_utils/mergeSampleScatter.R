mergeSampleScatter = function(data_list = list(repl.pairs_DPI7, repl.pairs_DP10,
                                               repl.pairs_1, repl.pairs_2),
                              data_type = c("DPI7, rep 1 vs rep 2", "DPI10, rep 1 vs rep 2",
                                            "rep 1, DPI7 vs DPI10", "rep 2, DPI7 vs DPI10")) {
  if(length(data_type) != length(data_list)) stop("data_list and data_type should be of the same length")
  
  # rename type column to match new labels in data_type
  for (i in seq_along(data_list)) {
    type_samples = data_list[[i]]$type_samples
    data_list[[i]]$repl.pairs[, type := gsub(type_samples, data_type[i], type)]
    data_list[[i]]$repl.labels[, type := gsub(type_samples, data_type[i], type)]
    data_list[[i]]$type_samples = data_type[i]
  }
  # rbind or c each element of the list
  ## 0. create a results environment
  res = new.env()
  ## 1. repl.pairs
  repl.pairs = lapply(data_list, function(x) x$repl.pairs)
  res$repl.pairs = rbindlist(repl.pairs)
  ### set order of panels for repl.pairs
  order_of_panels1 = vapply(data_type, function(data_type1) {
    grep(data_type1, unique(res$repl.pairs$type), value = TRUE)
  }, FUN.VALUE = character(1L))
  res$repl.pairs[, type := factor(type,
                                  levels = order_of_panels1)]
  ## 2. repl.labels
  repl.labels = lapply(data_list, function(x) x$repl.labels)
  res$repl.labels = rbindlist(repl.labels)
  ### set order of panels for repl.labels  
  order_of_panels2 = vapply(data_type, function(data_type1) {
    grep(data_type1, unique(res$repl.labels$type), value = TRUE)
  }, FUN.VALUE = character(1L))
  res$repl.labels[, type := factor(type,
                                   levels = order_of_panels2)]
  ## 3. all other elements which are just character(1L) 
  names_to_iter = names(data_list[[1]])
  names_to_iter = names_to_iter[!names_to_iter %in% c("repl.pairs", "repl.labels")]
  for (name in names_to_iter) {
    val = vapply(data_list, function(x) x[name][[1]], FUN.VALUE = character(1L))
    if(uniqueN(val) == 1) val = unique(val)
    assign(name, value = val, envir = res)
  }
  as.list(res)
}