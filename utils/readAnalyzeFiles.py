import pandas as pd
import os
import matplotlib.pyplot as plt
import numpy as np
import json


folderName = '/Users/valerioipem/Dropbox/MyIPEM/My_Projects/ArtScienceLab/Projects/CrossFitSonification/MAX/data/'
userFolders = os.listdir(folderName)
#old header
# header=['Timestamp','current_spine','neutral_spine','max_deviation_spine','current_barpath','initial_bar_position','day','month','year','hour','minutes','seconds','millis']

### new data format

header=['Timestamp','current_spine','neutral_spine','max_deviation_spine','distortion spine','current_barpath','initial_bar_position','panning map','right channel','left channel','day','month','year','hour','minutes','seconds','millis']

list_of_dfs = []
list_of_names = []

neutralSpine = {}
maxSpineBend = {}
initialBarbell = {}

data = {}

for userNr,name in enumerate(userFolders):
    if os.path.isdir(os.path.join(folderName,name)):
        userNrFolder = os.path.join(folderName,name)
        feedbacks = os.listdir(userNrFolder)

        data[name]={}
        
        for num,feedback in enumerate(feedbacks):
            if os.path.isdir(os.path.join(userNrFolder,feedback)):
                performancePoints = os.listdir(os.path.join(userNrFolder, feedback))
                data[name][feedback]={}

                for ppoint,performancePoint in enumerate(performancePoints):
                    if os.path.isdir(os.path.join(userNrFolder, feedback,performancePoint)):
                        fileNames = os.listdir(os.path.join(userNrFolder,feedback,performancePoint))

                        data[name][feedback][performancePoint] = []
                        
                        for fileName in fileNames:

                            df = pd.read_csv(os.path.join(userNrFolder,feedback,performancePoint,fileName), sep=';', names=header, index_col=False)
                            list_of_dfs.append(df)
                            nameTest = "%s_%s_%s" % (name,feedback,performancePoint)
                            list_of_names.append(nameTest)
                            data[name][feedback][performancePoint].append(nameTest)
                            data[name][feedback][performancePoint].append(df)


# calculate refences

for usrNr,userData in enumerate(data):
    neutralSpine[userData] = []
    if data[userData]['Control']['Combination'][1]['neutral_spine'][1000] != 1.:
        neutralSpine[userData].append(data[userData]['Control']['Combination'][1]['neutral_spine'][1000])
    else:
        neutralSpine[userData].append(data[userData]['Sonification']['Combination'][1]['neutral_spine'][1000])

    initialBarbell[userData] = []
    if data[userData]['Control']['Combination'][1]['initial_bar_position'][1000] != 1.:
        initialBarbell[userData].append(data[userData]['Control']['Combination'][1]['initial_bar_position'][1000])
    else:
        initialBarbell[userData].append(data[userData]['Sonification']['Combination'][1]['initial_bar_position'][1000])

    maxSpineBend[userData] = []
    if data[userData]['Control']['Combination'][1]['max_deviation_spine'][1000] != 999999:
        maxSpineBend[userData].append(data[userData]['Control']['Combination'][1]['max_deviation_spine'][1000])
    else:
        maxSpineBend[userData].append(data[userData]['Sonification']['Combination'][1]['max_deviation_spine'][1000])


print(neutralSpine)
print(maxSpineBend)
print(initialBarbell)

plotNr =0

for usrNr,userData in enumerate(data):

    for fdbnr,fdb in enumerate(data[userData]):

# spine plots

            plotNr=plotNr+1
            currSpine = data[userData][fdb]['Combination'][1]['current_spine']
            timeStamp = data[userData][fdb]['Combination'][1]['Timestamp']
            fig = plt.figure(plotNr)
            ax1 = fig.add_subplot(111)
            ax1.set_title(data[userData][fdb]['Combination'][0])
            ax1.set_xlabel('Time [ms]')
            ax1.set_ylabel('Spine markers distances [mm]')
            ax1.plot(timeStamp, currSpine, color='k')
            ax1.plot(timeStamp, np.ones(len(timeStamp)) * neutralSpine[userData], color='r')
            ax1.plot(timeStamp, np.ones(len(timeStamp)) * maxSpineBend[userData], color='b')

## barbell path plots
            plotNr = plotNr + 1

            currBarPos = data[userData][fdb]['Combination'][1]['current_barpath']
            timeStamp = data[userData][fdb]['Combination'][1]['Timestamp']
            fig = plt.figure(plotNr)
            ax1 = fig.add_subplot(111)
            ax1.set_title(data[userData][fdb]['Combination'][0])
            ax1.set_xlabel('Time [ms]')
            ax1.set_ylabel('Bar-foot distance [mm]')
            ax1.plot(timeStamp, currBarPos, color='k')
            ax1.plot(timeStamp, np.ones(len(timeStamp)) * initialBarbell[userData], color='c')



plt.show()