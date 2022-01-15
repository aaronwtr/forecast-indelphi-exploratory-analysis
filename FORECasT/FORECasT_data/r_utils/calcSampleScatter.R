calcSampleScatter = function(data_repl,
                             split_by = "complex_indel_classes",
                             value.var = "Norm_MCI_Reads",
                             x_group = "ST_June_2017_K562_800x_LV7A_DPI7",
                             y_group = "ST_June_2017_K562_800x_LV7B_DPI7",
                             x_name = "replicate 1", y_name = "replicate 2",
                             data_pred = NULL, sample_group_sel = "K562",
                             pred_group = "Predicted",
                             type_samples = "Replicates", type_predicted = "Predicted") {
  # process data for replicates / individual samples
  data_repl = data_repl[sample_group == sample_group_sel]
  data_repl[sample_ID == x_group, sample_group := x_name]
  data_repl[sample_ID == y_group, sample_group := y_name]
  repl = unique(data_repl[sample_group %in% c(x_name, y_name),
                          .(Oligo_Id, Most_Common_Indel, mean_Norm_MCI_Reads = get(value.var),
                            value.var = get(value.var), split_by = get(split_by), sample_group)])
  setnames(repl, c("value.var", "split_by"), c(value.var, split_by))
  
  repl.pairs = dcast.data.table(repl,
                                eval(parse(text = paste0("Oligo_Id + Most_Common_Indel + `",
                                                         split_by,"` ~ sample_group"))),
                                value.var = value.var, fill = 0)
  repl.pairs[, type := type_samples]
  
  # process data for predictions
  if(!is.null(data_pred)){
    data_pred = data_pred[sample_group %in% c(sample_group_sel, pred_group)]
    data_pred[sample_group == sample_group_sel, sample_group := paste0("Observed / ", x_name)]
    data_pred[sample_group == pred_group, sample_group := paste0("Predicted / ", y_name)]
    predicted.pairs = dcast.data.table(data_pred, Oligo_Id + Most_Common_Indel + few_indel_classes ~ sample_group,
                                       value.var = "Norm_MCI_Reads", fill = 0)
    predicted.pairs[, type := type_predicted]
    # rename samples in replicates
    setnames(repl.pairs, c(x_name, y_name),
             c(paste0("Observed / ", x_name), paste0("Predicted / ", y_name)))
    x_name = paste0("Observed / ", x_name)
    y_name = paste0("Predicted / ", y_name)
    # merge predictions are replicates
    repl.pairs = rbind(predicted.pairs, repl.pairs)
  }
  
  repl.pairs[, cor := cor(get(x_name), get(y_name)), by = .(type)]
  repl.pairs[, type := paste0(type, ", R: ", signif(cor, 2))]
  repl.pairs[, cor_by_class := cor(get(x_name), get(y_name)), by = .(get(split_by), type)]
  repl.pairs[, cor_label := paste0("R: ", signif(cor_by_class, 2))]
  repl.labels = unique(repl.pairs[, .(x = 0.35, y = 0.8, type,
                                      split_by = get(split_by), cor_label)])
  setnames(repl.labels, c("split_by"), c(split_by))
  list(repl.pairs = repl.pairs, repl.labels = repl.labels, 
       split_by = split_by, value.var = value.var,
       x_group = x_group, y_group = y_group,
       x_name = x_name, y_name = y_name,
       sample_group_sel = sample_group_sel, pred_group = pred_group,
       type_samples = type_samples, type_predicted = type_predicted)
}