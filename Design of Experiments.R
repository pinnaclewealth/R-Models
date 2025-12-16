rm(list=ls())
set.seed(123)


if(!require(FrF2)) install.packages("FrF2")
library(FrF2)


# 10 Binary features (2 level: -1 = don't include, +1 = include)
feat_names <- c(
  "LargeYard", "HardwoodFloors", "SmartThermostat", "FinishedBasement", "OpenKitchen",
  "Fireplace", "CornerLot", "View", "HomeOffice", "SolarRoof"
)

# 16 run fractional factorial for 10 factors
doe <- FrF2(nruns = 16, nfactors = 10, factor.names = feat_names, randomize = FALSE)

# view design
doe


# Design quality
class(doe)
design.info(doe)
summary(doe)


# Breaks aliases and allows clean estimation of main effects
base_df <- as.data.frame(doe)[ , feat_names, drop = FALSE]

#Combined object (32 rows: base and foldover)
doe_fold <- fold.design(doe, onefactors = NULL)
full_fold_df <- as.data.frame(doe_fold)[ , feat_names,  drop = FALSE] # drops any extra cols from foldover

# Extract rows 17-32
fold_df <- full_fold_df[-(1:nrow(base_df)), ,drop=FALSE] 

nrow(base_df); nrow(fold_df)

# combine the rows with the fold -- 32 runs (16+16)
doe32 <- rbind(base_df, fold_df) 

rownames(doe32) <- NULL

# keep a tag so we know which half is which
doe32$Part <- c(rep("Base", nrow(base_df)), rep("Foldover", nrow(fold_df)))

# Inspect/Save
head(doe32, 3)





