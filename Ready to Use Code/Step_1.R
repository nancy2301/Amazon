
#title: "Amazon new file input"
#author: "Eirik, Jillian, Nancy, Yifan, Ziyu "


# Prepare the environment 
knitr::opts_chunk$set(
  fig.width = 8.5,
  message = FALSE,
  warning = FALSE,
  # include = FALSE,
  echo = FALSE
)
options(tinytex.verbose = TRUE)

## Clear environment of variables and functions
rm(list = ls(all = TRUE)) 

## Clear environment of packages
if(is.null(sessionInfo()$otherPkgs) == FALSE) {
  lapply(paste("package:", names(sessionInfo()$otherPkgs), sep=""), 
         detach, character.only = TRUE, unload = TRUE)
}

## Load packages
library(tidyverse)
library(skimr)
library(sqldf)
library(gridExtra)
library(fpp2)
library(olsrr)
library(GGally)
library(lubridate)
library(kableExtra)
library(janitor)


# Load files

##!!!Change the adoption and billing file names for your files
new_file <- read.csv(file = "adoption_final.csv", header=TRUE, sep=",")
new_month <- read.csv(file = "june_billing_final.csv", header=TRUE, sep=",")




# Data cleaning

## Set factors 
new_file$Geo_Code <- as.character(new_file$Geo_Code)
new_file$Customer_size <- as.factor(new_file$Customer_size)
new_file$Month <- as.factor(new_file$Month)
new_file$Customer_ID <- as.factor(new_file$Customer_ID)

## If Geo_code is NA, set it as Geo_unclaimed
## Doing it this way because any other way does not remove 10 NAs 
new_file <- new_file %>% 
  mutate(Geo_Code =  case_when(Geo_Code == "AMER" ~ "AMER",
                               Geo_Code == "EMEA" ~ "EMEA",
                               Geo_Code == "APAC" ~ "APAC",
                               Geo_Code == "JAPN" ~ "JAPN",
                               Geo_Code == "CHNA" ~ "CHNA",
                               Geo_Code == "GLBL" ~ "GLBL",
                               TRUE ~ "GEO-UNCLAIMED"))

## Removing Geo Unclaimd, since only 60 some rows
new_file <- new_file %>%
  filter(Geo_Code != "GEO-UNCLAIMED")


## Calculating customer's age in months based on registration date and storing in a new column on each table
elapsed_months <- function(end_date, start_date) {
  ed <- as.POSIXlt(end_date)
  sd <- as.POSIXlt(start_date)
  12 * (ed$year - sd$year) + (ed$mon - sd$mon)
}
new_month$customer_age <- elapsed_months( new_month$Billing_month, new_month$Registration_date)

# Replace null product_id with 0
new_month <- sqldf('Select Billing_month, Customer_ID, Registration_date, 
      case when product_id is NULL then 0 else product_id end as product_id,                   Billed_amount, customer_age
               from new_month')

## Finding total number of product used for each customer
new_month_finish <- new_month %>%
  group_by(Customer_ID) %>%
  summarise(ProductsUsed = n(), totalBilled = sum(Billed_amount), customer_age = mean(customer_age)) 


## Join table customer ID as primary key
new_file_month <- new_file %>%
  filter(grepl("April",Month))
All <- new_file_month %>% left_join(new_month_finish , c("Customer_ID" = "Customer_ID"))

All$Customer_size <- as.factor(All$Customer_size)


## Drop remaining 7 NAs
All <- All %>%
  drop_na(ProductsUsed)



# Writing final dataset to CSV for Machine Learning in Python
write.csv(All, "new.csv")

