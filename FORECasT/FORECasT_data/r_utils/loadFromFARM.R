loadFromFARM = function(farm_dir, local_dir, n_files, which_files = " | grep \\.txt", ls_options = "", which_farm = "farm3-login", gzip = T) {
  if(dir.exists(local_dir) & isTRUE(length(list.files(local_dir)) == n_files)){
    data.files = paste0(local_dir,list.files(local_dir))
  } else {
    data.files = system(paste0("ssh ",which_farm," ls ",ls_options,farm_dir, which_files), intern = T)
    dir.create(local_dir, recursive = T)
    sapply(data.files, function(data.file)
      system(paste0("rsync ",which_farm,":",
                    farm_dir, data.file," ",
                    local_dir, " -r -av --delete")))
    data.files = paste0(local_dir,list.files(local_dir))
    data.files = data.files[!grepl("\\.gz", data.files)]
    if(isTRUE(gzip)){
      sapply(data.files, function(data.file)
        gzip(filename = data.file, destname = paste0(data.file,".gz"), remove = T, overwrite = T))
    }
    data.files = paste0(local_dir,list.files(local_dir))
  }
  return(data.files)
}