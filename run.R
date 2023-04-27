
#########################################################
# OVERVIEW
#   This script generates the folder structure and run the specified code for the paper:
#     "Paper Name"
#   
# PROJECT STRUCTURE
#     |--Clout-Chasing.Rproj 
#     |--code/              # store code files
#     |--data/              # store raw data
#     |--doc/               # store relevant documents
#     |--log/               # store log files
#     |--processed/         # store processed data
#     |   |--intermediate/  # store temp data during process
#     |--results/            
#     |   |--figures/       # store result figures
#     |   |--tables/        # store result tables
#     |--run.R              # project runner
#     |--temp/              # temp files
#
# SOFTWARE REQUIREMENTS
#   Analyses run on MacOS using Stata version 17 and R-4.2.0
#     with tidyverse package installed
#
# TO PERFORM A CLEAN RUN, DELETE THE FOLLOWING TWO FOLDERS:
#   /processed/intermediate
#   /results
#########################################################


## Use pacman rather than baser to manage packages
if (system.file(package='pacman') == "") {
  install.packages("pacman")
}

## Load necessary packages (auto install if not exist)
pacman::p_load(tidyverse)
pacman::p_load(here)
pacman::p_load(RStata)
pacman::p_load(rio)
pacman::p_load(fs)

## Load user defined functions
source("code/_utils.R")

## Project root
i_am(path = "run.R")

## Stata options
options("RStata.StataPath" = "/Applications/Stata/StataSE.app/Contents/MacOS/stata-se")
options("RStata.StataVersion" = 17)


# 1. Clean Previous Output (if exist) -------------------------------------
if (dir_exists(here("processed", "intermediate"))) {
  dir_delete(here("processed", "intermediate"))
}
if (dir_exists(here("results"))) {
  dir_delete(here("results"))
}


# 2. Create Output Directories --------------------------------------------
dir_create(here("processed", "intermediate"))
dir_create(here("results"))
dir_create(here("results", "figures"))
dir_create(here("results", "tables"))


# 3. Generate Main Data ---------------------------------------------------


# 4. Run Analysis ---------------------------------------------------------
run_script(path = "code/04_basic-analysis.R")
run_script(path = "code/temp.do")


### EOF
