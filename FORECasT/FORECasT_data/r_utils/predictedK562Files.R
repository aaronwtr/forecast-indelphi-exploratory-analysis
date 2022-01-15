predictedFiles = function(data.files, sample_shared_name = NULL, predicted = "Predicted", DPI = "DPI7") {
  DPI7.data.files = grep(DPI, data.files,value = T)
  # new
  new.DPI7.data.files = grep("LV7|NB|NA", DPI7.data.files, value = T)
  if(!is.null(sample_shared_name)){
    # choose 1 cell line
    new.DPI7.data.files = grep(sample_shared_name, new.DPI7.data.files, value = T) 
  } else {
    # discard no Cas9 cells 
    new.DPI7.data.files = grep("WT_12NA_DPI", new.DPI7.data.files, value = T, invert = T)
    # add RPE
    new.DPI7.data.files = c(new.DPI7.data.files, 
                            grep("ST_Feb_2018_RPE1_500x_7B_DPI7_dec", data.files, value = T))
  }
  
  # discard poor quality samples 
  new.DPI7.data.files = c(grep("RPE1_LV7|ST_June_2017_K562_1600x_LV7B_DPI7",
                               new.DPI7.data.files, value = T, invert = T),
                          grep(predicted, data.files, value = T))
  sort(new.DPI7.data.files)
}