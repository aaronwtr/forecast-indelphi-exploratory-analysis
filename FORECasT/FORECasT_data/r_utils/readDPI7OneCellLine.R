readOneCellLine = function(data.files, sample_shared_name, DPI = 7){
  DPI = paste0("DPI", DPI)
  DPI7.data.files = grep(DPI, data.files,value = T)
  # choose 1 cell line
  DPI7.data.files = grep(sample_shared_name, DPI7.data.files, value = T) 
  # new
  new.DPI7.data.files = grep("LV7|NB|NA", DPI7.data.files,value = T)
  # discard poor quality samples ST_Feb_2018_RPE1_500x_7B_DPI7_dec
  new.DPI7.data.files = grep("RPE1_LV7|ST_June_2017_K562_1600x_LV7B_DPI7",
                             new.DPI7.data.files, value = T, invert = T)
  new.DPI7.data.files = sort(new.DPI7.data.files)
}