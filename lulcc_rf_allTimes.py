# Colin Doyle
# 2020/06/02

# Run the random forest algorithm with defined number of trees and variables per node and assess the accuracy with the
# original 10 classes, then with the forest degradation and agriculture classes combined, then with the final 8 classes
# used in the study (also combining the 2 forest types). The 9 class results were not reported in the manuscript.

# It then runs this for all 3 time periods in the study.
# The results will save to json files, one file for each time period.

import json
import datetime
import ee
ee.Initialize()

startTime = datetime.datetime.now()
# import stuff
ow = ee.FeatureCollection('users/csddoyle/orange_walk')
alos = ee.Image("JAXA/ALOS/AW3D30/V2_2")
l8 = ee.ImageCollection("LANDSAT/LC08/C01/T1_SR")
l5 = ee.ImageCollection("LANDSAT/LT05/C01/T1_SR")
l7 = ee.ImageCollection("LANDSAT/LE07/C01/T1_SR")
l4 = ee.ImageCollection("LANDSAT/LT04/C01/T1_SR")
trainFC = ee.FeatureCollection('users/csddoyle/Dissertation/LULCC/trainFC_2015_10class_final')
valFC = ee.FeatureCollection('users/csddoyle/Dissertation/LULCC/valFC_2015_10class_final')
# Variables per split for RF
varNode = 3
# Number of trees in RF
trees = 300
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


# # MAKE NDVI TIME SERIES BY DOY FOR MULTIPLE YEARS AVE TO FIGURE OUT SEASONAL COMPOSITES filter to Orange Walk
# owL8 = l8.filterBounds(ow).filterDate('2014-01-01', '2016-12-31')
def clipFn(img):
    return img.clip(ow)


# slope and intercept citation: Roy, D.P., Kovalskyy, V., Zhang, H.K., Vermote, E.F., Yan, L., Kumar, S.S, Egorov, A.,
# 2016, Characterization of Landsat - 7 to Landsat - 8 reflective wavelength and normalized difference vegetation index
# continuity, Remote Sensing of Environment, 185, 57 - 70.(http: // dx.doi.org / 10.1016 / j.rse .2015 .12 .024) Table
# 2 - reduced major axis(RMA) regression coefficients
# harmonize oli to tm
# def oli2tm(oli):
#     slopes = ee.Image.constant([0.9785, 0.9542, 0.9825, 1.0073, 1.0171, 0.9949])
#     itcp = ee.Image.constant([-0.0095, -0.0016, -0.0022, -0.0021, -0.0030, 0.0029])
#     y = oli.select(['B2', 'B3', 'B4', 'B5', 'B6', 'B7'], ['b', 'g', 'r', 'nir', 'swir1', 'swir2'])\
#         .subtract(itcp.multiply(10000)).divide(slopes) \
#         .set('system:time_start', oli.get('system:time_start'))
#     return y.toShort()


# harmonize tm and etm+ to oli
def tm2oli(tm):
    slopes = ee.Image.constant([0.9785, 0.9542, 0.9825, 1.0073, 1.0171, 0.9949]);
    itcp = ee.Image.constant([-0.0095, -0.0016, -0.0022, -0.0021, -0.0030, 0.0029]);
    y = tm.select(['B1','B2','B3','B4','B5','B7'],['b', 'g', 'r', 'nir', 'swir1', 'swir2']).resample('bicubic')\
        .multiply(slopes).add(itcp.multiply(10000)).set('system:time_start', tm.get('system:time_start'))\
        .set({'system:index':tm.get('system:index')})
    return y.toShort().copyProperties(tm)


def ndwiFn(img):
    return img.normalizedDifference(['nir', 'swir1'])


def ndviFn(img):
    return img.normalizedDifference(['nir', 'r'])


# Dry images
ow2014 = l8.filterBounds(ow).filterDate('2014-02-01', '2014-07-31')
ow2015 = l8.filterBounds(ow).filterDate('2015-02-01', '2015-07-31')
ow2016 = l8.filterBounds(ow).filterDate('2016-02-01', '2016-07-31')
dry = ow2014.merge(ow2015).merge(ow2016)
dryNoCloud = dry.map(maskClouds).select(['B2', 'B3', 'B4', 'B5', 'B6', 'B7'], ['b', 'g', 'r', 'nir', 'swir1', 'swir2'])
# dry = dry.map(clipFn)
# Harmonize L8 to L5 so we can apply same algorithm to other years
dryl8harm = dryNoCloud  #.map(oli2tm)

# Wet images
owF2014 = l8.filterBounds(ow).filterDate('2014-08-01', '2015-01-31')
owF2015 = l8.filterBounds(ow).filterDate('2015-08-01', '2016-01-31')
owF2016 = l8.filterBounds(ow).filterDate('2016-08-01', '2017-01-31')
wet = owF2014.merge(owF2015).merge(owF2016)
wetNoCloud = wet.map(maskClouds).select(['B2', 'B3', 'B4', 'B5', 'B6', 'B7'], ['b', 'g', 'r', 'nir', 'swir1', 'swir2'])
# wet = wet.map(clipFn)
# Harmonize L8 to L5 so we can apply same algorithm to other years
wetl8harm = wetNoCloud  #.map(oli2tm)

# DRY SEASON VARIABLES
# Dry Median
dryMed = dryl8harm.median().convolve(ee.Kernel.octagon(2))
dryNDVI = dryl8harm.map(ndviFn)
dryMedNDVI = dryNDVI.median().rename(['ndvi_d'])
dryNDWI = dryl8harm.map(ndwiFn)
dryMedNDWI = dryNDWI.median().rename(['ndwi_d'])
dryBands = dryMed.select(['b', 'g', 'r', 'nir', 'swir1', 'swir2'], ['b_d', 'g_d', 'r_d', 'nir_d', 'swir1_d', 'swir2_d'])

# WET SEASON VARIABLES
# Wet Median
wetMed = wetl8harm.median().convolve(ee.Kernel.octagon(2))
wetNDVI = wetl8harm.map(ndviFn)
wetMedNDVI = wetNDVI.median().rename(['ndvi_w'])
wetNDWI = wetl8harm.map(ndwiFn)
wetMedNDWI = wetNDWI.median().rename(['ndwi_w'])
wetBands = wetMed.select(['b', 'g', 'r', 'nir', 'swir1', 'swir2'], ['b_w', 'g_w', 'r_w', 'nir_w', 'swir1_w', 'swir2_w'])

# -------- NOW TRAIN AND BUILD THE ALGO! --------------
# turn training polygons into an image to sample
trainImg = trainFC.reduceToImage(['class'], ee.Reducer.mode()).rename(['class'])
# stack variables for classification
stack = ee.Image(dryBands.addBands(wetBands).addBands(dryMedNDVI).addBands(dryMedNDWI).addBands(wetMedNDVI)
                 .addBands(wetMedNDWI).addBands(dem).addBands(slope).addBands(tpi_500).addBands(trainImg)).clip(ow)

bands = ['b_d', 'g_d', 'r_d', 'nir_d', 'swir1_d', 'swir2_d', 'b_w', 'g_w', 'r_w', 'nir_w', 'swir1_w', 'swir2_w',
        'ndvi_d', 'ndwi_d', 'ndvi_w', 'ndwi_w', 'dsm', 'slope', 'tpi_500']

# To train with certain number of pixels per class
training = stack.stratifiedSample(numPoints=450, classBand='class', scale=30)

# validation data from separate set of polygons
valImg = valFC.reduceToImage(['class'], ee.Reducer.mode()).rename(['valClass'])
valStack = stack.select(bands).addBands(valImg).clip(ow)
validation = valStack.stratifiedSample(numPoints=450, classBand='valClass', scale=30)

# make a random forest classifier and train it
randForest = ee.Classifier.smileRandomForest(numberOfTrees=trees, variablesPerSplit=varNode, bagFraction=0.33) \
    .train(features=training, classProperty='class', inputProperties=bands)

# run the classification
classified = stack.select(bands).classify(randForest)

# Get a confusion matrix representing re-substitution accuracy.
trainAccuracy = randForest.confusionMatrix()
info = randForest.explain()
# Get validation accuracy estimate
validated = validation.classify(randForest)
valAccuracy = validated.errorMatrix('valClass', 'classification')

results2015 = 'results/final_2015_results_300-3.json'
colNames = ['Run', 'NumTrees', 'VarPerNode', 'ResubMatrix', 'TrainAcc', 'Importance', 'OOB_Error', 'Val_Acc',
            'Val_Matrix']
df = {'runs': []}
df['runs'].append({'Run': 'classes_10', 'Data': {'NumTrees': trees,
                                                 'VarPerNode': varNode,
                                                 'ResubMatrix': trainAccuracy.getInfo(),
                                                 'TrainAcc': trainAccuracy.accuracy().getInfo(),
                                                 'Importance': info.get('importance').getInfo(),
                                                 'OOB_Error': info.get('outOfBagErrorEstimate').getInfo(),
                                                 'Val_Acc': valAccuracy.accuracy().getInfo(),
                                                 'Val_Matrix': valAccuracy.getInfo()}})

# Calculate with fresh_cut and agriculture combined
# validation data from separate set of polygons
valFC9 = ee.FeatureCollection("users/csddoyle/Dissertation/LULCC/valFC_2015_9class_ag_final")
valImg9 = valFC9.reduceToImage(['class'], ee.Reducer.mode()).rename(['valClass'])
valStack9 = stack.select(bands).addBands(valImg9).clip(ow)
validation9 = valStack9.stratifiedSample(numPoints=450, classBand='valClass', scale=30)
validated9 = validation9.classify(randForest)

def remap9Class(ft):
    newClass = ft.set({'classification': ee.Algorithms.If(ee.Algorithms.IsEqual(ft.get('classification'), 10), 1,
                                                          ft.get('classification'))})
    return newClass


remap9Val = validated9.map(remap9Class)
valAccuracy9 = remap9Val.errorMatrix('valClass', 'classification')
# append the results to the json
df['runs'].append({'Run': 'classes_9', 'Data': {'NumTrees': trees,
                                                'VarPerNode': varNode,
                                                'ResubMatrix': trainAccuracy.getInfo(),
                                                'TrainAcc': trainAccuracy.accuracy().getInfo(),
                                                'Importance': info.get('importance').getInfo(),
                                                'OOB_Error': info.get('outOfBagErrorEstimate').getInfo(),
                                                'Val_Acc': valAccuracy9.accuracy().getInfo(),
                                                'Val_Matrix': valAccuracy9.getInfo()}})

# Calculate with fresh_cut and agriculture combined
# validation data from separate set of polygons
valFC8 = ee.FeatureCollection("users/csddoyle/Dissertation/LULCC/valFC_2015_8class_final")
valImg8 = valFC8.reduceToImage(['class'], ee.Reducer.mode()).rename(['valClass'])
valStack8 = stack.select(bands).addBands(valImg8).clip(ow)
validation8 = valStack8.stratifiedSample(numPoints=450, classBand='valClass', scale=30)
validated8 = validation8.classify(randForest)

def remap8Class(ft):
    newClass = ft.set({'classification': ee.Algorithms.If(ee.Algorithms.IsEqual(ft.get('classification'), 10), 1,
                                                          ft.get('classification'))})
    newValClass = newClass.set({'classification': ee.Algorithms.If(ee.Algorithms.IsEqual(newClass.get('classification'),
                                                                                         9), 5,
                                                                   newClass.get('classification'))})
    return newValClass


remap8Val = validated8.map(remap8Class)
valAccuracy8 = remap8Val.errorMatrix('valClass', 'classification')
df['runs'].append({'Run': 'classes_8', 'Data': {'NumTrees': trees,
                                                'VarPerNode': varNode,
                                                'ResubMatrix': trainAccuracy.getInfo(),
                                                'TrainAcc': trainAccuracy.accuracy().getInfo(),
                                                'Importance': info.get('importance').getInfo(),
                                                'OOB_Error': info.get('outOfBagErrorEstimate').getInfo(),
                                                'Val_Acc': valAccuracy8.accuracy().getInfo(),
                                                'Val_Matrix': valAccuracy8.getInfo()}})

with open(results2015, 'w') as outfile:
    json.dump(df, outfile)


export15 = ee.batch.Export.image.toDrive(image=classified,
                                       description='2015_classification',
                                       fileNamePrefix='rf_classification_2015',
                                       fileFormat='GeoTIFF',
                                       scale=30,
                                       maxPixels=1e13)
export15.start()

# ---------------------- 1999-2001 --------------------------------

def maskStripes(img):
    b1mask = img.select('B1').gte(0)
    b2mask = img.select('B2').gte(0)
    b3mask = img.select('B3').gte(0)
    b4mask = img.select('B4').gte(0)
    b5mask = img.select('B5').gte(0)
    b7mask = img.select('B7').gte(0)
    stripeMask = ee.Image(b1mask.add(b2mask).add(b3mask).add(b4mask).add(b5mask).add(b7mask)).eq(6)
    return img.updateMask(stripeMask).copyProperties(img).set({'system:time_start': img.get('system:time_start')})\
        .set({'system:index': img.get('system:index')})


# Dry images
ow99 = l5.filterBounds(ow).filterDate('1999-02-01', '1999-07-31')
ow00 = l5.filterBounds(ow).filterDate('2000-02-01', '2000-07-31')
ow01 = l5.filterBounds(ow).filterDate('2001-02-01', '2001-07-31')
owl799 = l7.filterBounds(ow).filterDate('1999-02-01', '1999-07-31')
owl700 = l7.filterBounds(ow).filterDate('2000-02-01', '2000-07-31')
owl701 = l7.filterBounds(ow).filterDate('2001-02-01', '2001-07-31')
dry00 = ow99.merge(ow00).merge(ow01).merge(owl799).merge(owl700).merge(owl701)
dryMask00 = dry00.map(maskStripes)
dryNoCloud00 = dryMask00.map(maskClouds)
# Harmonize L8 to L5 so we can apply same algorithm to other years
dryl8harm00 = dryNoCloud00.map(tm2oli)

# Wet images
owF99 = l5.filterBounds(ow).filterDate('1999-08-01', '2000-01-31')
owF00 = l5.filterBounds(ow).filterDate('2000-08-01', '2001-01-31')
owF01 = l5.filterBounds(ow).filterDate('2001-08-01', '2002-01-31')
owl7F99 = l7.filterBounds(ow).filterDate('1999-08-01', '2000-01-31')
owl7F00 = l7.filterBounds(ow).filterDate('2000-08-01', '2001-01-31')
owl7F01 = l7.filterBounds(ow).filterDate('2001-08-01', '2002-01-31')
wet00 = owF99.merge(owF00).merge(owF01).merge(owl7F99).merge(owl7F00).merge(owl7F01)
wetMask00 = wet00.map(maskStripes)
wetNoCloud00 = wetMask00.map(maskClouds)
# Harmonize L8 to L5 so we can apply same algorithm to other years
wetl8harm00 = wetNoCloud00.map(tm2oli)

# DRY SEASON VARIABLES
# Dry Median
dryMed00 = dryl8harm00.median().convolve(ee.Kernel.octagon(2))
dryNDVI00 = dryl8harm00.map(ndviFn)
dryMedNDVI00 = dryNDVI00.median().rename(['ndvi_d'])
dryNDWI00 = dryl8harm00.map(ndwiFn)
dryMedNDWI00 = dryNDWI00.median().rename(['ndwi_d'])
dryBands00 = dryMed00.select(['b', 'g', 'r', 'nir', 'swir1', 'swir2'], ['b_d', 'g_d', 'r_d', 'nir_d', 'swir1_d', 'swir2_d'])

# WET SEASON VARIABLES
# Wet Median
wetMed00 = wetl8harm00.median().convolve(ee.Kernel.octagon(2))
wetNDVI00 = wetl8harm00.map(ndviFn)
wetMedNDVI00 = wetNDVI00.median().rename(['ndvi_w'])
wetNDWI00 = wetl8harm00.map(ndwiFn)
wetMedNDWI00 = wetNDWI00.median().rename(['ndwi_w'])
wetBands00 = wetMed00.select(['b', 'g', 'r', 'nir', 'swir1', 'swir2'], ['b_w', 'g_w', 'r_w', 'nir_w', 'swir1_w', 'swir2_w'])

# stack variables for classification
stack00 = ee.Image(dryBands00.addBands(wetBands00).addBands(dryMedNDVI00).addBands(dryMedNDWI00).addBands(wetMedNDVI00)
                 .addBands(wetMedNDWI00).addBands(dem).addBands(slope).addBands(tpi_500)).clip(ow)

classified00 = stack00.select(bands).classify(randForest)
print('classified 2000')

# Get validation accuracy estimate
valFC00 = ee.FeatureCollection('users/csddoyle/Dissertation/LULCC/valFC_2000_10class_final')
valImg00 = valFC00.reduceToImage(['class'], ee.Reducer.mode()).rename(['valClass'])
valStack00 = stack00.select(bands).addBands(valImg00).clip(ow)
validation00 = valStack00.stratifiedSample(numPoints=450, classBand='valClass', scale=30)
validated00 = validation00.classify(randForest)
valAccuracy00 = validated00.errorMatrix('valClass', 'classification')

results2000 = 'results/final_2000_results_300-3.json'
colNames = ['Run', 'NumTrees', 'VarPerNode', 'ResubMatrix', 'TrainAcc', 'Importance', 'OOB_Error', 'Val_Acc',
            'Val_Matrix']
df00 = {'runs': []}
df00['runs'].append({'Run': 'classes_10', 'Data': {'NumTrees': trees,
                                                 'VarPerNode': varNode,
                                                 'ResubMatrix': trainAccuracy.getInfo(),
                                                 'TrainAcc': trainAccuracy.accuracy().getInfo(),
                                                 'Importance': info.get('importance').getInfo(),
                                                 'OOB_Error': info.get('outOfBagErrorEstimate').getInfo(),
                                                 'Val_Acc': valAccuracy00.accuracy().getInfo(),
                                                 'Val_Matrix': valAccuracy00.getInfo()}})

# Calculate with fresh_cut and agriculture combined
# validation data from separate set of polygons
valFC009 = ee.FeatureCollection("users/csddoyle/Dissertation/LULCC/valFC_2000_9class__ag_final")
valImg009 = valFC009.reduceToImage(['class'], ee.Reducer.mode()).rename(['valClass'])
valStack009 = stack00.select(bands).addBands(valImg009).clip(ow)
validation009 = valStack009.stratifiedSample(numPoints=450, classBand='valClass', scale=30)
validated009 = validation009.classify(randForest)

remap009Val = validated009.map(remap9Class)
valAccuracy009 = remap009Val.errorMatrix('valClass', 'classification')
# append the results to the json
df00['runs'].append({'Run': 'classes_9', 'Data': {'NumTrees': trees,
                                                'VarPerNode': varNode,
                                                'ResubMatrix': trainAccuracy.getInfo(),
                                                'TrainAcc': trainAccuracy.accuracy().getInfo(),
                                                'Importance': info.get('importance').getInfo(),
                                                'OOB_Error': info.get('outOfBagErrorEstimate').getInfo(),
                                                'Val_Acc': valAccuracy009.accuracy().getInfo(),
                                                'Val_Matrix': valAccuracy009.getInfo()}})

# Calculate with fresh_cut and agriculture combined
# validation data from separate set of polygons
valFC008 = ee.FeatureCollection("users/csddoyle/Dissertation/LULCC/valFC_2000_8class_final")
valImg008 = valFC008.reduceToImage(['class'], ee.Reducer.mode()).rename(['valClass'])
valStack008 = stack00.select(bands).addBands(valImg008).clip(ow)
validation008 = valStack008.stratifiedSample(numPoints=450, classBand='valClass', scale=30)
validated008 = validation008.classify(randForest)

remap8Val00 = validated008.map(remap8Class)
valAccuracy008 = remap8Val00.errorMatrix('valClass', 'classification')
df00['runs'].append({'Run': 'classes_8', 'Data': {'NumTrees': trees,
                                                'VarPerNode': varNode,
                                                'ResubMatrix': trainAccuracy.getInfo(),
                                                'TrainAcc': trainAccuracy.accuracy().getInfo(),
                                                'Importance': info.get('importance').getInfo(),
                                                'OOB_Error': info.get('outOfBagErrorEstimate').getInfo(),
                                                'Val_Acc': valAccuracy008.accuracy().getInfo(),
                                                'Val_Matrix': valAccuracy008.getInfo()}})

with open(results2000, 'w') as outfile00:
    json.dump(df00, outfile00)


export00 = ee.batch.Export.image.toDrive(image=classified00,
                                       description='2000_classification',
                                       fileNamePrefix='rf_classification_2000',
                                       fileFormat='GeoTIFF',
                                       scale=30,
                                       maxPixels=1e13)
export00.start()

# ---------------------- 1984-1987 --------------------------------
# Dry images
ow84 = l5.filterBounds(ow).filterDate('1984-02-01', '1984-07-31')
ow85 = l5.filterBounds(ow).filterDate('1985-02-01', '1985-07-31')
ow86 = l5.filterBounds(ow).filterDate('1986-02-01', '1986-07-31')
ow87 = l5.filterBounds(ow).filterDate('1987-02-01', '1987-07-31')
owl484 = l4.filterBounds(ow).filterDate('1984-02-01', '1984-07-31')
owl485 = l4.filterBounds(ow).filterDate('1985-02-01', '1985-07-31')
owl486 = l4.filterBounds(ow).filterDate('1986-02-01', '1986-07-31')
owl487 = l4.filterBounds(ow).filterDate('1987-02-01', '1987-07-31')
dry85 = ow84.merge(ow85).merge(ow86).merge(ow87).merge(owl484).merge(owl485).merge(owl486).merge(owl487)
dryMask85 = dry85.map(maskStripes)
dryNoCloud85 = dryMask85.map(maskClouds)
# Harmonize L8 to L5 so we can apply same algorithm to other years
dryl8harm85 = dryNoCloud85.map(tm2oli)

# Wet images
owF84 = l5.filterBounds(ow).filterDate('1984-08-01', '1985-01-31')
owF85 = l5.filterBounds(ow).filterDate('1985-08-01', '1986-01-31')
owF86 = l5.filterBounds(ow).filterDate('1986-08-01', '1987-01-31')
owF87 = l5.filterBounds(ow).filterDate('1987-08-01', '1988-01-31')
owl4F84 = l4.filterBounds(ow).filterDate('1984-08-01', '1985-01-31')
owl4F85 = l4.filterBounds(ow).filterDate('1985-08-01', '1986-01-31')
owl4F86 = l4.filterBounds(ow).filterDate('1986-08-01', '1987-01-31')
owl4F87 = l4.filterBounds(ow).filterDate('1987-08-01', '1988-01-31')
wet85 = owF84.merge(owF85).merge(owF86).merge(owF87).merge(owl4F84).merge(owl4F85).merge(owl4F86).merge(owl4F87)
wetMask85 = wet85.map(maskStripes)
wetNoCloud85 = wetMask85.map(maskClouds)
# Harmonize L8 to L5 so we can apply same algorithm to other years
wetl8harm85 = wetNoCloud85.map(tm2oli)

# DRY SEASON VARIABLES
# Dry Median
dryMed85 = dryl8harm85.median().convolve(ee.Kernel.octagon(2))
dryNDVI85 = dryl8harm85.map(ndviFn)
dryMedNDVI85 = dryNDVI85.median().rename(['ndvi_d'])
dryNDWI85 = dryl8harm85.map(ndwiFn)
dryMedNDWI85 = dryNDWI85.median().rename(['ndwi_d'])
dryBands85 = dryMed85.select(['b', 'g', 'r', 'nir', 'swir1', 'swir2'], ['b_d', 'g_d', 'r_d', 'nir_d', 'swir1_d', 'swir2_d'])

# WET SEASON VARIABLES
# Wet Median
wetMed85 = wetl8harm85.median().convolve(ee.Kernel.octagon(2))
wetNDVI85 = wetl8harm85.map(ndviFn)
wetMedNDVI85 = wetNDVI85.median().rename(['ndvi_w'])
wetNDWI85 = wetl8harm85.map(ndwiFn)
wetMedNDWI85 = wetNDWI85.median().rename(['ndwi_w'])
wetBands85 = wetMed85.select(['b', 'g', 'r', 'nir', 'swir1', 'swir2'], ['b_w', 'g_w', 'r_w', 'nir_w', 'swir1_w', 'swir2_w'])

# stack variables for classification
stack85 = ee.Image(dryBands85.addBands(wetBands85).addBands(dryMedNDVI85).addBands(dryMedNDWI85).addBands(wetMedNDVI85)
                 .addBands(wetMedNDWI85).addBands(dem).addBands(slope).addBands(tpi_500)).clip(ow)

classified85 = stack85.select(bands).classify(randForest)
print('classified 1985')

# Get validation accuracy estimate
valFC85 = ee.FeatureCollection('users/csddoyle/Dissertation/LULCC/valFC_1985_10class_final')
valImg85 = valFC85.reduceToImage(['class'], ee.Reducer.mode()).rename(['valClass'])
valStack85 = stack85.select(bands).addBands(valImg85).clip(ow)
validation85 = valStack85.stratifiedSample(numPoints=450, classBand='valClass', scale=30)
validated85 = validation85.classify(randForest)
valAccuracy85 = validated85.errorMatrix('valClass', 'classification')

results1985 = 'results/final_1985_results_300-3.json'
colNames = ['Run', 'NumTrees', 'VarPerNode', 'ResubMatrix', 'TrainAcc', 'Importance', 'OOB_Error', 'Val_Acc',
            'Val_Matrix']
df85 = {'runs': []}
df85['runs'].append({'Run': 'classes_10', 'Data': {'NumTrees': trees,
                                                 'VarPerNode': varNode,
                                                 'ResubMatrix': trainAccuracy.getInfo(),
                                                 'TrainAcc': trainAccuracy.accuracy().getInfo(),
                                                 'Importance': info.get('importance').getInfo(),
                                                 'OOB_Error': info.get('outOfBagErrorEstimate').getInfo(),
                                                 'Val_Acc': valAccuracy85.accuracy().getInfo(),
                                                 'Val_Matrix': valAccuracy85.getInfo()}})

# Calculate with fresh_cut and agriculture combined
# validation data from separate set of polygons
valFC859 = ee.FeatureCollection("users/csddoyle/Dissertation/LULCC/valFC_1985_9class_ag_final")
valImg859 = valFC859.reduceToImage(['class'], ee.Reducer.mode()).rename(['valClass'])
valStack859 = stack85.select(bands).addBands(valImg859).clip(ow)
validation859 = valStack859.stratifiedSample(numPoints=450, classBand='valClass', scale=30)
validated859 = validation859.classify(randForest)

remap859Val = validated859.map(remap9Class)
valAccuracy859 = remap859Val.errorMatrix('valClass', 'classification')
# append the results to the json
df85['runs'].append({'Run': 'classes_9', 'Data': {'NumTrees': trees,
                                                'VarPerNode': varNode,
                                                'ResubMatrix': trainAccuracy.getInfo(),
                                                'TrainAcc': trainAccuracy.accuracy().getInfo(),
                                                'Importance': info.get('importance').getInfo(),
                                                'OOB_Error': info.get('outOfBagErrorEstimate').getInfo(),
                                                'Val_Acc': valAccuracy859.accuracy().getInfo(),
                                                'Val_Matrix': valAccuracy859.getInfo()}})

# Calculate with fresh_cut and agriculture combined
# validation data from separate set of polygons
valFC858 = ee.FeatureCollection("users/csddoyle/Dissertation/LULCC/valFC_1985_8class_final")
valImg858 = valFC858.reduceToImage(['class'], ee.Reducer.mode()).rename(['valClass'])
valStack858 = stack85.select(bands).addBands(valImg858).clip(ow)
validation858 = valStack858.stratifiedSample(numPoints=450, classBand='valClass', scale=30)
validated858 = validation858.classify(randForest)

remap8Val85 = validated858.map(remap8Class)
valAccuracy858 = remap8Val85.errorMatrix('valClass', 'classification')
df85['runs'].append({'Run': 'classes_8', 'Data': {'NumTrees': trees,
                                                'VarPerNode': varNode,
                                                'ResubMatrix': trainAccuracy.getInfo(),
                                                'TrainAcc': trainAccuracy.accuracy().getInfo(),
                                                'Importance': info.get('importance').getInfo(),
                                                'OOB_Error': info.get('outOfBagErrorEstimate').getInfo(),
                                                'Val_Acc': valAccuracy858.accuracy().getInfo(),
                                                'Val_Matrix': valAccuracy858.getInfo()}})

with open(results1985, 'w') as outfile85:
    json.dump(df85, outfile85)


export85 = ee.batch.Export.image.toDrive(image=classified85,
                                         description='1985_classification',
                                         fileNamePrefix='rf_classification_1985',
                                         fileFormat='GeoTIFF',
                                         scale=30,
                                         maxPixels=1e13)
export85.start()
