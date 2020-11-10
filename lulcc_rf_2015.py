# Colin Doyle
# 2020/4/7
#
# Use Random Forest for mapping LULC in Belize. This builds the model using Landsat 8 data and tests a range of values
# for the number of trees and number of variables per split parameters. It will save the training accuracy, validation
# accuracy, OOB error, and confusion matrix for each run as a CSV.


import pandas as pd
import datetime
import ee
ee.Initialize()

# save the results here
result_path = "results/all2015_results.csv"
# list all trees and var/node we want to test
treeList = [1, 25, 50, 100, 150, 200, 250, 300, 350, 400, 450, 500, 550, 600]
varNodeList = [1, 2, 3, 4, 5]


# Time the run
startTime = datetime.datetime.now()
# import stuff
ow = ee.FeatureCollection('users/csddoyle/orange_walk')
alos = ee.Image("JAXA/ALOS/AW3D30/V2_2")
l8 = ee.ImageCollection("LANDSAT/LC08/C01/T1_SR")
# Training and validation datasets
trainFC = ee.FeatureCollection('users/csddoyle/Dissertation/LULCC/trainFC_2015_10class_final')
valFC = ee.FeatureCollection('users/csddoyle/Dissertation/LULCC/valFC_2015_10class_final')

# Orange Walk ROI
ow = ow.geometry()

# DEM
dem = alos.select(['AVE_DSM'], ['dsm']).clip(ow)
# slope from alos
slope = ee.Terrain.slope(dem)
# Weiss et al.(http: // www.jennessent.com / downloads / tpi - poster - tnc_18x22.pdf) use an annulus.
kernel = ee.Kernel.circle(500, 'meters', False).add(ee.Kernel.circle(300, 'meters', False, -1))
tpi_500 = dem.subtract(dem.focal_mean(kernel=kernel)).rename(['tpi_500'])

# Cloud masking algo
def maskClouds(image):
    cloud_pixel_qa = image.select('pixel_qa').bitwiseAnd(32).eq(0)
    cloud_shadow_pixel_qa = image.select('pixel_qa').bitwiseAnd(8).eq(0)
    imageSansCloudandShadow = image.updateMask(cloud_pixel_qa.And(cloud_shadow_pixel_qa))
    return ee.Image(imageSansCloudandShadow).copyProperties(image).set({'system:time_start': image.get('system:time_start')})

def clipFn(img):
    return img.clip(ow)

# slope and intercept citation: Roy, D.P., Kovalskyy, V., Zhang, H.K., Vermote, E.F., Yan, L., Kumar, S.S, Egorov, A.,
# 2016, Characterization of Landsat - 7 to Landsat - 8 reflective wavelength and normalized difference vegetation index
# continuity, Remote Sensing of Environment, 185, 57 - 70.(http: // dx.doi.org / 10.1016 / j.rse .2015 .12 .024) Table
# 2 - reduced major axis(RMA) regression coefficients
# harmonize oli to tm
def oli2tm(oli):
    slopes = ee.Image.constant([0.9785, 0.9542, 0.9825, 1.0073, 1.0171, 0.9949])
    itcp = ee.Image.constant([-0.0095, -0.0016, -0.0022, -0.0021, -0.0030, 0.0029])
    y = oli.select(['B2', 'B3', 'B4', 'B5', 'B6', 'B7'], ['b', 'g', 'r', 'nir', 'swir1', 'swir2'])\
        .subtract(itcp.multiply(10000)).divide(slopes) \
        .set('system:time_start', oli.get('system:time_start'))
    return y.toShort()


# // harmonize tm and etm+ to oli
# var tm2oli = function(tm) {
#   var slopes = ee.Image.constant([0.9785, 0.9542, 0.9825, 1.0073, 1.0171, 0.9949]);
#   var itcp = ee.Image.constant([-0.0095, -0.0016, -0.0022, -0.0021, -0.0030, 0.0029]);
#    var y = tm.select(['B1','B2','B3','B4','B5','B7'],['b', 'g', 'r', 'nir', 'swir1', 'swir2'])
#              .resample('bicubic')
#              .multiply(slopes).add(itcp.multiply(10000))
#              .set('system:time_start', tm.get('system:time_start')).set({'system:index':tm.get('system:index')});
#   return y.toShort().copyProperties(tm);
# };

# Dry images
ow2014 = l8.filterBounds(ow).filterDate('2014-02-01', '2014-07-31')
ow2015 = l8.filterBounds(ow).filterDate('2015-02-01', '2015-07-31')
ow2016 = l8.filterBounds(ow).filterDate('2016-02-01', '2016-07-31')
dry = ow2014.merge(ow2015).merge(ow2016)
dryNoCloud = dry.map(maskClouds).select(['B2', 'B3', 'B4', 'B5', 'B6', 'B7'], ['b', 'g', 'r', 'nir', 'swir1', 'swir2'])
# Harmonize L8 to L5 so we can apply same algorithm to other years
dryl8harm = dryNoCloud   # .map(oli2tm)

# Wet images
owF2014 = l8.filterBounds(ow).filterDate('2014-08-01', '2015-01-31')
owF2015 = l8.filterBounds(ow).filterDate('2015-08-01', '2016-01-31')
owF2016 = l8.filterBounds(ow).filterDate('2016-08-01', '2017-01-31')
wet = owF2014.merge(owF2015).merge(owF2016)
wetNoCloud = wet.map(maskClouds).select(['B2', 'B3', 'B4', 'B5', 'B6', 'B7'], ['b', 'g', 'r', 'nir', 'swir1', 'swir2'])
# Harmonize L8 to L5 so we can apply same algorithm to other years
wetl8harm = wetNoCloud   # .map(oli2tm)

# DRY SEASON VARIABLES
# Dry Median
dryMed = dryl8harm.median().convolve(ee.Kernel.octagon(2))
# Dry NDVI collection
def ndviFn(img):
    return img.normalizedDifference(['nir', 'r'])
dryNDVI = dryl8harm.map(ndviFn)

# Dry NDVI Median
dryMedNDVI = dryNDVI.median().rename(['ndvi_d'])
# Dry NDWI collection
def ndwiFn(img):
    return img.normalizedDifference(['nir', 'swir1'])
dryNDWI = dryl8harm.map(ndwiFn)
# Dry NDWI Median
dryMedNDWI = dryNDWI.median().rename(['ndwi_d'])
# Extract Dry bands for classification
dryBands = dryMed.select(['b', 'g', 'r', 'nir', 'swir1', 'swir2'], ['b_d', 'g_d', 'r_d', 'nir_d', 'swir1_d', 'swir2_d'])

# WET SEASON VARIABLES
# Wet Median
wetMed = wetl8harm.median().convolve(ee.Kernel.octagon(2))
# Wet NDVI collection
wetNDVI = wetl8harm.map(ndviFn)
# Wet NDVI Median
wetMedNDVI = wetNDVI.median().rename(['ndvi_w'])
# Wet NDWI collection
wetNDWI = wetl8harm.map(ndwiFn)
# Wet NDWI Median
wetMedNDWI = wetNDWI.median().rename(['ndwi_w'])
# Extract Wet bands for classification
wetBands = wetMed.select(['b', 'g', 'r', 'nir', 'swir1', 'swir2'], ['b_w', 'g_w', 'r_w', 'nir_w', 'swir1_w', 'swir2_w'])

# -------- NOW TRAIN AND BUILD THE ALGO! --------------
# turn training polygons into an image to sample
trainImg = trainFC.reduceToImage(['class'], ee.Reducer.mode()).rename(['class'])
# stack variables for classification
stack = ee.Image(dryBands.addBands(wetBands).addBands(dryMedNDVI).addBands(dryMedNDWI).addBands(wetMedNDVI)
                 .addBands(wetMedNDWI).addBands(dem).addBands(slope).addBands(tpi_500).addBands(trainImg)).clip(ow)


bands =['b_d', 'g_d', 'r_d', 'nir_d', 'swir1_d', 'swir2_d', 'b_w', 'g_w', 'r_w', 'nir_w', 'swir1_w', 'swir2_w',
        'ndvi_d', 'ndwi_d', 'ndvi_w', 'ndwi_w', 'dsm', 'slope', 'tpi_500']

# To train with all pixels in polygons
# training = stack.select(bands).sampleRegions(collection=trainFC, properties=['code'], scale=30)

# To train with certain number of pixels per class
training = stack.stratifiedSample(numPoints=450, classBand='class', scale=30)

# create a csv to save all of the results in
all2015Results = result_path

colNames = ['Run', 'NumTrees', 'VarPerNode', 'ResubMatrix', 'TrainAcc', 'Importance', 'OOB_Error', 'Val_Acc']
runs = []
for i in range(1, 71, 1):
    runs.append('run_{}'.format(i))
numTrees = range(1, 71, 1)
varPerNode = range(1, 71, 1)
resubMatrix = range(1, 71, 1)
trainAcc = range(1, 71, 1)
importance = range(1, 71, 1)
oob = range(1, 71, 1)
valAcc = range(1, 71, 1)
df = pd.DataFrame({'Run': runs,
                   'NumTrees': numTrees,
                   'VarPerNode': varPerNode,
                   'ResubMatrix': resubMatrix,
                   'TrainAcc': trainAcc,
                   'Importance': importance,
                   'OOB_Error': oob,
                   'Val_Acc': valAcc})
df = df.astype({'Run': str, 'ResubMatrix': object, 'TrainAcc': float, 'Importance': object, 'OOB_Error': float,
                'Val_Acc': float})
df.to_csv(all2015Results, index=False, columns=colNames)

# ------------- run RF with all 70 parameter combinations and save outputs -------------
# validation data from separate set of polygons
valImg = valFC.reduceToImage(['class'], ee.Reducer.mode()).rename(['valClass'])
valStack = stack.select(bands).addBands(valImg).clip(ow)
validation = valStack.stratifiedSample(numPoints=450, classBand='valClass', scale=30)

runNum = 1
for trees in treeList:
    for varNode in varNodeList:
        run = 'run_{}'.format(runNum)
        # make a random forest classifier and train it
        randForest = ee.Classifier.smileRandomForest(numberOfTrees=trees, variablesPerSplit=varNode, bagFraction=0.33)\
            .train(features=training, classProperty='class', inputProperties=bands)

        # run the classification
        classified = stack.select(bands).classify(randForest)

        # Get a confusion matrix representing re-substitution accuracy.
        trainAccuracy = randForest.confusionMatrix()
        print(run)
        print('Resubstitution error matrix: {}'.format(trainAccuracy.getInfo()))
        print('Training overall accuracy: {}'.format(trainAccuracy.accuracy().getInfo()))

        info = randForest.explain()
        print('importance: {}'.format(info.get('importance').getInfo()))
        print('numTree: {}'.format(info.get('numberOfTrees').getInfo()))
        print('OOB error: {}'.format(info.get('outOfBagErrorEstimate').getInfo()))

        # Get validation accuracy estimate
        validated = validation.classify(randForest)
        valAccuracy = validated.errorMatrix('valClass', 'classification')
        print('validation error matrix: {}'.format(valAccuracy.getInfo()))
        print('validation overall accuracy: {}'.format(valAccuracy.accuracy().getInfo()))

        print('------------------------------------------------------------')

        # append the results to the csv
        resultCsv = pd.read_csv(all2015Results, index_col='Run', dtype={'ResubMatrix': object, 'TrainAcc': float,
                                                                        'Importance': object, 'OOB_Error': float,
                                                                        'Val_Acc': float})
        # resultCsv = resultCsv.astype({'Run': int, 'ResubMatrix': object, 'TrainAcc': float, 'Importance': object, 'OOB_Error': float})
        resultCsv.loc[run, 'NumTrees'] = info.get('numberOfTrees').getInfo()
        resultCsv.loc[run, 'VarPerNode'] = varNode
        resultCsv.at[run, 'ResubMatrix'] = trainAccuracy.getInfo()
        resultCsv.loc[run, 'TrainAcc'] = trainAccuracy.accuracy().getInfo()
        resultCsv.at[run, 'Importance'] = info.get('importance').getInfo()
        resultCsv.loc[run, 'OOB_Error'] = info.get('outOfBagErrorEstimate').getInfo()
        resultCsv.loc[run, 'Val_Acc'] = valAccuracy.accuracy().getInfo()
        resultCsv.to_csv(all2015Results)

        runNum = runNum + 1


endTime = datetime.datetime.now()
runTime = endTime - startTime
print('Run time: {}'.format(runTime))

