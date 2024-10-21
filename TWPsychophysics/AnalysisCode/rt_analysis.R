# quick check ds 
# 2024-10-09/18

library(tidyverse)
library(readxl) # if dealing with XLS data
library(janitor)
library(gt)

# Get participant details and parameters
PartInitials = "RH"
folder <- str_glue("../Data/{PartInitials}Data/")

# regex for finding files
filepattern <- str_glue("^TW_{PartInitials}_.*.xlsx")

# get all files
files = list.files(folder, pattern = filepattern, full.names = TRUE )

# make a function for reading in a file
read_psychopy_data <- function(f) {
  read_excel(f) %>% 
    clean_names()
}

# then can use idea of map (or loops if preferred)
# to go through data and process



# in this section - assuming TIDY data format... --------------------------
# i tidied on example file

# regex for finding files
filepattern_csv <- str_glue("^TW_{PartInitials}_.*.csv")

# get all files
files_csv = list.files(folder, pattern = filepattern_csv, full.names = TRUE )

# load data
d <- read_csv(files_csv[1])

# fix the parenthesis issue ['right'] -> right
# idea: extract the word left or right ..if present and
#       overwrite the direction column  
d <- d %>%
  mutate(direction = str_extract(direction, 'left|right')) %>% 
  mutate(contrast = round(contrast, digits = 2))  %>% 
  rowid_to_column(var='trial_no')
# rounding is necessary... as contrast has some floating point madness
# 0.6000002 or similar shenanigans from psychopy / python.

# now group and summarise

# summary
s <- d %>% 
  group_by(direction, contrast) %>% 
  summarise(mean_travel_time = mean(time),
            std_travel_time = sd(time)) 

# a table...
gt(s)

# a plot
# NOTE... trial order is randomised in original data... so should get that back
#.        therefore here, it looks bunched up into 3rds
d %>% 
  ggplot(aes(x = trial_no, y = time, color = direction)) +
  geom_point() +
  geom_line() +
  facet_wrap(~contrast, ncol = 1) +
  theme_minimal() +
  scale_color_brewer(palette = "Set1") +
  scale_y_continuous(limits = c(0, 4))
  
