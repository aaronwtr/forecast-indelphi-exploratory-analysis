renameSampleGroup = function(data){
  data[sample_group == "K562_800x",  sample_group := "K562"]
  data[sample_group == "CAS9_12NA",  sample_group := "K562"]
  data[sample_group == "BOB",  sample_group := "Human iPSC"]
  data[sample_group == "E14TG2A",  sample_group := "Mouse ESC"]
  data[sample_group == "RPE1_500x_7B",  sample_group := "RPE1"]
  data
} 

