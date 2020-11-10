# owbz_lulcc
Author: Colin Doyle
2020/11/10

Analysis for land-use land-cover change study from Doyle et al., 2020. These scripts were used to generate a random forest algorithm to map land-use and land-cover in Orange Walk, Belize for 3 time periods using Google Earth Engine.

`lulcc_rf_2015.py` runs the experiment to optimize the random forest number of trees and number of variable per split parameters. Results saved as a csv containing all of the accuracy metrics and confusion matrices for every combination of parameters.

`lulcc_rf_allTimes.py` runs the random forest algorithm with a defined number of trees and variables per split. This runs the final results used in the paper, classifying all 3 time periods and saving results for 10 classes, 9 classes, and 8 classes. The 9 class results are not mentioned in the manuscript. The results are saved as json files, 1 file for each time period contained the accruacy metrics and confusion matrices for all 3 class combinations. 

`rf_graphs.py` contains the code used to generate the sensitivity analysis plot in the manuscript as well as the NDVI and precipitation time series graphs in the manuscript.

All of the csv and json files are the results exported from these Python scripts, as well as organized into various ways for plotting the results. 

You can recreate the environment used to run these scripts with the packages in the requirements.txt file.
