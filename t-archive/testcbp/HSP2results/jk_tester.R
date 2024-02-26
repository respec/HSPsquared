#!/usr/bin/R
library("rhdf5") #https://bioc.ism.ac.jp/packages/3.4/bioc/vignettes/rhdf5/inst/doc/rhdf5.pdf

# Find and load the h5 file
# dir()

# specify .h5 file
h5_file_name = "PL3_5250_0001.h5" 

# open the file in R
h5f = H5Fopen(h5_file_name)
# output the keys within the h5 file
h5f

# h5read(h5_file_name, "/CONTROL/EXT_SOURCES/table") # open an entire table
# h5read(h5_file_name, "/CONTROL/EXT_SOURCES/table")[1:10,] # open 10 rows of a table
# h5read(h5_file_name, "RESULTS/RCHRES_R001/HYDR/table")[1:10,]

# content.df <- h5ls(h5_file_name,all=TRUE)[1:10,]
content.df <- h5ls(h5_file_name,all=TRUE)
colnames(content.df)

# investigating TIMESERIES group
ts_group.df <- content.df[grep("TIMESERIES", content.df$group), ]
ts_group.df <- content.df[grep("TIMESERIES/TS", content.df$group), ]
ts_group.df <- content.df[grep("values", content.df$name), ]
ts_group.df[1:20,]
tail(ts_group.df)
length(ts_group.df[,1])
# h5read(h5_file_name, "/TIMESERIES/TS011/table")[1:10,]
# TS_table <- h5read(h5_file_name, "/TIMESERIES/TS011/table")
TS_table <- h5read(h5_file_name, "/TIMESERIES/TS011/table")
# TS_table <- h5read(h5_file_name, "/TIMESERIES/RCHRES_R001/table")
TS_table[1:10,]
tail(TS_table)

# investigating RESULTS group
# ts_group.df <- content.df[grep("RESULTS/RCHRES_R001/HYDR", content.df$group), ]
# ts_group.df[1:20,]
# tail(ts_group.df)
# length(ts_group.df[,1])

# helpful printouts
# HYDR_table <- h5read(h5_file_name, "RESULTS/RCHRES_R001/HYDR/table")
# HYDR_table[1:10,]
# tail(HYDR_table)
# length(HYDR_table[,1])
# write.csv(HYDR_table, "HYDR_table_vsetOUTDGT.csv")