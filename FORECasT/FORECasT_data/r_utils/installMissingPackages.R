installMissingPackages = function(packages){
  packages_ind = packages %in% names(installed.packages()[,"Package"])
  if(mean(packages_ind) != 1){
    packages_to_install = packages[!packages %in% names(installed.packages()[,"Package"])]
    # specifying mirror is necessary for some Linux systems
    install.packages(packages_to_install, dependencies = T, repos = "http://mirrors.ebi.ac.uk/CRAN/")
    packages_to_install = packages[!packages %in% names(installed.packages()[,"Package"])]
    source("https://bioconductor.org/biocLite.R")
    biocLite()
    biocLite(packages_to_install, dependencies = T)
  }
  # show which packages were installed
  if(mean(packages_ind) != 1) installed.packages()[names(installed.packages()[,"Package"]) %in% packages[!packages_ind],]
}