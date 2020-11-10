# Colin Doyle
# 2020-4-7

# Let's make some graphs from the results!

import numpy as np
import matplotlib.pylab as plt
import pandas as pd

import calendar

def JulianDate_to_MMDD(y, jd):
    month = 1
    day = 0
    while jd - calendar.monthrange(y,month)[1] > 0 and month <= 12:
        jd = jd - calendar.monthrange(y,month)[1]
        month = month + 1
    return "{}-{}".format(month, jd)

file = pd.read_csv('all2015_results_20200602.csv')
var1 = pd.read_csv('sensitivity_1var.csv')
var5 = pd.read_csv('sensitivity_5var.csv')
tree25 = pd.read_csv('sensitivity_25trees.csv')
tree50 = pd.read_csv('sensitivity_50trees.csv')
tree100 = pd.read_csv('sensitivity_100trees.csv')
tree200 = pd.read_csv('sensitivity_200trees.csv')
tree300 = pd.read_csv('sensitivity_300trees.csv')
tree400 = pd.read_csv('sensitivity_400trees.csv')
tree500 = pd.read_csv('sensitivity_500trees.csv')
tree600 = pd.read_csv('sensitivity_600trees.csv')

# print(var1['OOB_Error'])
oobAcc_1var = [(1-x)*100 for x in var1['OOB_Error']]
oobAcc_5var = [(1-x)*100 for x in var5['OOB_Error']]
var1_acc = [x*100 for x in var1['Val_Acc']]
var5_acc = [x*100 for x in var5['Val_Acc']]

plt.rcParams.update({'font.size': 12, 'font.sans-serif': 'Arial'})


# Combined sensitivity plots
fig, (ax1, ax2) = plt.subplots(1, 2)

ax1.plot(var1['NumTrees'], oobAcc_1var, label='OOB accuracy (m=1)', marker='.', color='black', markersize=10)
ax1.plot(var1['NumTrees'], var1_acc, label='validation accuracy (m=1)', marker='.', color='gray', markersize=10)
ax1.plot(var5['NumTrees'], oobAcc_5var, label='OOB accuracy (m=5)', marker='^', color='black', markersize=10, linestyle='--')
ax1.plot(var5['NumTrees'], var5_acc, label='validation accuracy (m=5)', marker='^', color='gray', markersize=10, linestyle='--')
ax1.set_xlabel('Number of Trees', labelpad=15)
ax1.set_ylabel('Accuracy (%)', labelpad=15)
ax1.set_ylim(80)
ax1.set_xticks(range(0, 650, 50))
# ax1.tick_params(labelrotation=45)
ax1.set_xticklabels(var1['NumTrees'], rotation=45)
ax1.legend(bbox_to_anchor=(1, 0), loc=4, frameon=True, fontsize=16, facecolor='white')
# plt.title('Effect of Number of Trees on Accuracy', pad=10)
ax1.grid()
# ax1.show()

tree25acc = [x*100 for x in tree25['Val_Acc']]
tree50acc = [x*100 for x in tree50['Val_Acc']]
tree100acc = [x*100 for x in tree100['Val_Acc']]
tree200acc = [x*100 for x in tree200['Val_Acc']]
tree300acc = [x*100 for x in tree300['Val_Acc']]
tree400acc = [x*100 for x in tree400['Val_Acc']]
tree500acc = [x*100 for x in tree500['Val_Acc']]
tree600acc = [x*100 for x in tree600['Val_Acc']]


ax2.plot(tree25['VarPerNode'], tree25acc, label='25 trees', marker='.', markersize=10)
ax2.plot(tree50['VarPerNode'], tree50acc, label='50 trees', marker='.', markersize=10)
ax2.plot(tree100['VarPerNode'], tree100acc, label='100 trees', marker='.', markersize=10)
ax2.plot(tree200['VarPerNode'], tree200acc, label='200 trees', marker='.', markersize=10)
ax2.plot(tree300['VarPerNode'], tree300acc, label='300 trees', marker='.', markersize=10)
ax2.plot(tree400['VarPerNode'], tree400acc, label='400 trees', marker='.', markersize=10)
ax2.plot(tree500['VarPerNode'], tree500acc, label='500 trees', marker='.', markersize=10)
ax2.plot(tree600['VarPerNode'], tree600acc, label='600 trees', marker='.', markersize=10)
ax2.set_xlabel('Number of Variables Per Node', labelpad=15)
ax2.set_ylabel('Accuracy (%)', labelpad=15)
ax2.set_ylim(86)
ax2.set_xticks(range(1, 6, 1))
ax2.legend(bbox_to_anchor=(1, 0), loc=4, frameon=True, fontsize=16, facecolor='white', ncol=2)
# plt.title('Effect of Variables per Node on Accuracy', pad=10)
ax2.grid()
fig.set_size_inches(12.5, 5, forward=True)
fig.tight_layout()
fig.show()
plt.savefig('combined_sensitivity.pdf', dpi=500)



# NDVI and precip plot
fig, (ax1, ax2) = plt.subplots(2)

ndvi_csv = pd.read_csv("ndvi_timeseries_trainFC.csv")
dayrange = np.arange(1, 366, 1)
# doy_labels = [JulianDate_to_MMDD(2015, day) for day in dayrange]
new_labels = [date for date in dayrange if (date-1) % 30 == 0]
doy_labels = [JulianDate_to_MMDD(2015, day) for day in new_labels]
# print(doy_labels)
print(list(ndvi_csv['doy'])[-1])
ax1.plot(ndvi_csv['doy'], ndvi_csv['agriculture'].fillna(method='ffill'), label="Agriculture", marker='.', color='#ffee01')
ax1.plot(ndvi_csv['doy'], ndvi_csv['rice'].fillna(method='ffill'), label="Rice", marker='.', color='#826800')
ax1.plot(ndvi_csv['doy'], ndvi_csv['savanna'].fillna(method='ffill'), label="Savanna/Shrubland", marker='.', color='#ed7b10')
ax1.plot(ndvi_csv['doy'], ndvi_csv['forest'].fillna(method='ffill'), label="Lowland broad-leaved moist forest", marker='.', color='#2dd736')
ax1.plot(ndvi_csv['doy'], ndvi_csv['scrub_forest'].fillna(method='ffill'), label="Lowland broad-leaved moist scrub forest", marker='.', color='#42871b')
ax1.plot(ndvi_csv['doy'], ndvi_csv['mangrove'].fillna(method='ffill'), label="Swamp forest", marker='.', color='#400080')
ax1.plot(ndvi_csv['doy'], ndvi_csv['wetland'].fillna(method='ffill'), label="Wetland", marker='.', color='#bb20ea')
ax1.plot(ndvi_csv['doy'], ndvi_csv['water'].fillna(method='ffill'), label="Water", marker='.', color='#003589')
ax1.plot(ndvi_csv['doy'], ndvi_csv['urban'].fillna(method='ffill'), label="Urban/Roads", marker='.', color='#de0a0a')
ax1.plot(ndvi_csv['doy'], ndvi_csv['fresh_cut'].fillna(method='ffill'), label="Forest degradation", marker='.', color='#7c0600')
ax1.set_xticks(np.arange(1, 366, 30))
ax1.set_xticklabels([])
ax1.set_xlim([1, 366])
# ax1.set_xlabel("Day of Year", labelpad=10)
ax1.set_ylabel("NDVI", labelpad=10)
ax1.grid()
ax1.legend(bbox_to_anchor=(1,1), loc=4, ncol=2, frameon=False, fontsize=12, facecolor='white')
# fig.show()

precip_csv = pd.read_csv("ow_gsmap_mean.csv")
ax2.bar(precip_csv['doy'], precip_csv['hourlyPrecipRateGC_mean'], color='blue')
ax2.set_xticks(np.arange(1, 366, 30))
ax2.set_xlim([1, 366])
ax2.set_xticklabels(doy_labels)
ax2.set_xlabel("Day of Year", labelpad=10)
ax2.set_ylabel("Mean daily precipitation (mm)", labelpad=10)
ax2.grid()
fig.set_size_inches(8, 6, forward=True)
fig.tight_layout()
fig.show()
# plt.savefig('figure2.png', dpi=300)


# Individual sensitivity plots.
# numVar, tree1, tree25, tree50, tree100, tree200, tree300, tree400,
# tree500 = np.loadtxt('rf2015_varNode_oob-error.csv', unpack=True, delimiter=',',skiprows=1)
# plt.plot(numVar, tree1, label='1 tree', marker='.')
# plt.plot(numVar, tree25, label='25 trees', marker='.')
# plt.plot(numVar, tree50, label='50 trees', marker='.')
# plt.plot(numVar, tree100, label='100 trees', marker='.')
# plt.plot(numVar, tree200, label='200 trees', marker='.')
# plt.plot(numVar, tree300, label='300 trees', marker='.')
# plt.plot(numVar, tree400, label='400 trees', marker='.')
# plt.plot(numVar, tree500, label='500 trees', marker='.')
#
# plt.xlabel('Number of Variables per Node', labelpad=10)
# plt.ylabel('Out-of-bag Error', labelpad=10)
# # plt.ylim(0.95)
# plt.legend(bbox_to_anchor=(1, 1), loc=2, frameon=True, fontsize=12, facecolor='white')
# plt.title('Effect of Trees and Variables per Node on Random Forest OOB Error', pad=10)
# plt.grid()
# plt.show()

#
# numVar, tree1, tree25, tree50, tree100, tree200, tree300, tree400, tree500 = np.loadtxt('rf2015_varNode_trainAccuracy.csv', unpack=True, delimiter=',',
#                                                     skiprows=1)
# # plt.plot(numVar, tree1, label='1 tree', marker='.')
# plt.plot(numVar, tree25, label='25 trees', marker='.')
# plt.plot(numVar, tree50, label='50 trees', marker='.')
# plt.plot(numVar, tree100, label='100 trees', marker='.')
# plt.plot(numVar, tree200, label='200 trees', marker='.')
# plt.plot(numVar, tree300, label='300 trees', marker='.')
# plt.plot(numVar, tree400, label='400 trees', marker='.')
# plt.plot(numVar, tree500, label='500 trees', marker='.')
#
# plt.xlabel('Number of Variables per Node', labelpad=10)
# plt.ylabel('Training Accuracy', labelpad=10)
# # plt.ylim(0.95)
# plt.legend(bbox_to_anchor=(1, 1), loc=2, frameon=True, fontsize=12, facecolor='white')
# plt.title('Effect of Trees and Variables per Node on Random Forest Training Accuracy', pad=10)
# plt.grid()
# plt.show()