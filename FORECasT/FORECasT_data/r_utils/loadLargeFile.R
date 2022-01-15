loadLargeFile = function(file = "gerp_conservation_scores.homo_sapiens.bw",
                         dir = "./data/",
                         url = "ftp://ftp.ensembl.org/pub/current_compara/conservation_scores/70_mammals.gerp_conservation_score/gerp_conservation_scores.homo_sapiens.bw",
                         md5_url = "ftp://ftp.ensembl.org/pub/current_compara/conservation_scores/70_mammals.gerp_conservation_score/MD5SUM"){
  file_path = paste0(dir, file)
  if(!dir.exists(dir)) dir.create(dir)
  if(!file.exists(file_path)){
    download.file(url, 
                  file_path)
    # check integrity of the download
    if(!is.null(md5_url)) {
      MD5SUM = readLines(md5_url)
      MD5SUM = grep(file, MD5SUM, value = T)
      MD5SUM = substr(MD5SUM, 1, 32)
      if (.Platform$pkgType == "mac.binary.el-capitan") {
        command = "md5"
      } else if (.Platform$OS.type == "unix") {
        command = "md5sum"
      }
      # digest file and compare MD5SUM
      md5_res = system2(command, file_path,
                        stdout = TRUE)
      if(command == "md5sum"){
        md5_res = substr(md5_res, 1, 32)
      } else {
        md5_res = substr(md5_res, nchar(md5_res)-31, nchar(md5_res))
      }
      if(MD5SUM != md5_res) stop("downloaded file is corrupt, please download again")
    }
  } else message("file already downloaded")
  file_path
}