code_path = "./"
results_path = "../results/"

# Figures that use only Cas9 in K562
rmarkdown::render(input = paste0(code_path, "Figures_K562.Rmd"), 
                  output_format = "html_document", 
                  output_file=paste0(results_path, "Figures_K562_analysis_report.html"))

# Figures that use Cas9-TREX and Cas9-2A-TREX
rmarkdown::render(input = paste0(code_path, "Figures_size_5E_S18.Rmd"), 
                  output_format = "html_document", 
                  output_file=paste0(results_path, "Figures_size_5E_S18_analysis_report.html"))

# Figures that use multiple cell lines and Cas9-2A-TREX
rmarkdown::render(input = paste0(code_path, "Figures_bins_5D_S17.Rmd"), 
                  output_format = "html_document", 
                  output_file=paste0(results_path, "Figures_bins_5D_S17_analysis_report.html"))

# Figures that include FORECasT model predictions
rmarkdown::render(input = paste0(code_path, "Figures_K562_predicted_FORECasT_v2.Rmd"), 
                  output_format = "html_document", 
                  output_file=paste0(results_path, "Figures_K562_predicted_FORECasT_v2_analysis_report.html"))

# Mapping gRNA target sites to protein domains to produce Figure S25
rmarkdown::render(input = paste0(code_path, "Figure_S25.Rmd"), 
                  output_format = "html_document", 
                  output_file=paste0(results_path, "Figure_S25_analysis_report.html"))
