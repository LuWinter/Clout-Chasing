
#########################################################
# SCRIPT: 01_detect-posts.R
# TASK: This script detects the event-relevant posts
#########################################################


# PACKAGES ----------------------------------------------------------------
pacman::p_load(dplyr)
pacman::p_load(arrow)
pacman::p_load(readxl)
pacman::p_load(lubridate)
pacman::p_load(purrr)
pacman::p_load(stringr)


# 1. Prepare Data ---------------------------------------------------------
## Open HuDongE dataset 
hde_arrow <- open_dataset(sources = "data/HuDongE/")
hde_arrow |> 
  count(Year) |> 
  collect() |> 
  arrange(Year)

## Fetch 2017-2022 HuDongE data
hde_data <- hde_arrow |> 
  filter(Year >= 2017) |> 
  collect()
hde_data |> 
  head() |> 
  print(width = Inf)

## Handle date problem
hde_data <- hde_data |> 
  filter(
    str_length(`提问时间`) >= 10
  ) |> 
  mutate(
    提问时间 = ymd(str_sub(`提问时间`, end = 10L)),
    Month = month(提问时间)
  )

## Generate post id
hde_data <- hde_data |> 
  group_by(Year, Month) |> 
  mutate(
    id = paste0(Year, str_pad(Month, 2, "left", "0")),
    id = paste0(id, str_pad(row_number(), 5, "left", "0"))
  )

## Load hot events data
hot_events <- read_excel(path = 'data/hot-events_20230325.xlsx')
hot_events$Date <- ymd(hot_events$Date)
head(hot_events, 10)


# 2. Conduct detecting ----------------------------------------------------
## Detect function
detect_post <- function(event, date) {
  start_date <- date - 15
  end_date <- date + 30
  
  res <- hde_data |> 
    filter(
      str_detect(提问内容, event),
      `提问时间` >= start_date,
      `提问时间` <= end_date
    )
  res
}

post_list <- map(
  .x = 1:nrow(hot_events), 
  .f = \(x) {
    res <- detect_post(
      event = hot_events[[x, 1]], 
      date = hot_events[[x, 4]]
    )
    res$Event <- hot_events[[x, 1]]
    res
  }
)

## Combine data and remove duplicate
post_df <- reduce(.x = post_list, .f = bind_rows) |>
  distinct(id, .keep_all = TRUE)

## Write result to csv
csv_path = "processed/detected-post_20230326.csv"
write_csv_arrow(x = post_df, sink = csv_path)


### EOF