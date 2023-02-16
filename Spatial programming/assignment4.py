import os
import pandas as pd
from datetime import datetime
import geopandas as gpd
import matplotlib.pyplot as mplt
from collections import Counter
import numpy as np
import fiona
fiona.supported_drivers



# pre-set all paths that will be used
shp_path2018 = os.path.join(os.path.curdir, "data", "VicRoadsAccidents", "2018", "2018.shp")
shp_path2017 = os.path.join(os.path.curdir, "data", "VicRoadsAccidents", "2017", "2017.shp")
shp_path2016 = os.path.join(os.path.curdir, "data", "VicRoadsAccidents", "2016", "2016.shp")
shp_path2015 = os.path.join(os.path.curdir, "data", "VicRoadsAccidents", "2015", "2015.shp")
shp_path2014 = os.path.join(os.path.curdir, "data", "VicRoadsAccidents", "2014", "2014.shp")
shp_path2013 = os.path.join(os.path.curdir, "data", "VicRoadsAccidents", "2013", "2013.shp")
path_arr = [shp_path2018, shp_path2017, shp_path2016, shp_path2015, shp_path2014, shp_path2013]

path_lga2017 = os.path.join(os.path.curdir, "data", "RegionsLGA_2017", "LGA_2017_VIC.shp")
path_sa2 = os.path.join(os.path.curdir, "data", "RegionsSA2_2016", "SA2_2016_AUST.shp")

# setting dataframe print limitation
pd.set_option('display.max_rows', None)
pd.set_option('display.max_columns', None)
pd.set_option('display.width', 250)



# print(os.path.exists(shp_path2015))
# file = gpd.read_file(shp_path2017)
# file.plot()
# mplt.show()
# print(shp_path2017)
'''4.1 start================================='''

def task411():

    records = []
    acctype_count = {}
    for e in path_arr:
        with fiona.open(e, 'r') as src:
            records.append(len(src))  # get num of records from all years
            for f in src:
                type_name = f['properties']['ACCIDENT_1']
                acctype_count[type_name] = acctype_count.get(type_name, 0) + 1
        src.close()

    avg = round(sum(records) / len(records), 2)
    print("(a)The average number of accidents per year is", avg)  # print 4.1.1, as required

    max_type_acc = max(acctype_count, key=acctype_count.get)  # max value
    pct_acc = acctype_count.get(max_type_acc) / sum(records) * 100

    snd_val = sorted(acctype_count.values())[-2]  # sort by value and get the 2nd last
    idx = list(acctype_count.values()).index(snd_val)
    snd_key = list(acctype_count.keys())[idx]
    pct_acc2 = snd_val / sum(records) * 100

    print(
        f"(b) The most common type of accident in all the recorded years is {max_type_acc}(equal to {round(pct_acc, 2)}% accidents),"
        f" and the second most common is {snd_key}({round(pct_acc2, 2)}%).")

    return None

task411()
'''4.1.2==========================================================================='''

def statistic_acc(src_path):
    """this function is for statistic of the number of accidents by
       LGA_NAME(column) in the sourcefile(one year)"""

    count_dict = {}
    with fiona.open(src_path, 'r') as src:
        for f in src:
            name = f['properties']['LGA_NAME']
            count_dict[name] = count_dict.get(name, 0) + 1
    src.close()
    return count_dict


def acci_by_year():
    """this function returns statistic of the number of accidents by
       LGA_NAME of all years(2013-2018)"""

    # retrieve data, count all accident numbers of each LGA in each year
    result_statistic = dict.fromkeys(['2018', '2017', '2016', '2015', '2014', '2013'])
    for e in path_arr:
        year = str(e.rsplit('\\', 2)[1])
        # start retrieve data
        tmp_dict = statistic_acc(e)
        result_statistic[year] = tmp_dict
    return result_statistic

def task412():

    # 4.2a started
    result_dict = dict.fromkeys(['2018','2017','2016','2015','2014','2013'])
    for e in path_arr:
        year = str(e.rsplit('\\',2)[1])

        # start retrieve data
        with fiona.open(e, 'r') as src:
            tmp_heavyv = []
            tmp_psgv = []
            tmp_motor = []
            tmp_publv = []
            for f in src:
                tmp_heavyv.append(f['properties']['HEAVYVEHIC'])
                tmp_psgv.append(f['properties']['PASSENGERV'])
                tmp_motor.append(f['properties']['MOTORCYCLE'])
                tmp_publv.append(f['properties']['PUBLICVEHI'])

        result_dict[year] = [sum(tmp_heavyv),sum(tmp_psgv),sum(tmp_motor),sum(tmp_publv)]
        src.close()

        new_df = pd.DataFrame(result_dict)  # convert to dataframe
        new_df = new_df.rename(index={0:"Heavy_vhc",1:"Passenger_vhc",2:"Motorcycle",3:"Public_vhc"})
        pd.set_option('display.max_rows', 10)
        pd.set_option('display.max_columns', 10)

    print(new_df)  # print table as 4.2a required


    # 4.2b started
    yearAccOccur = acci_by_year()
    # lets see the 2013 data, and sort it to find top 10
    dict2013 = yearAccOccur['2013']
    dict2013 = {k: v for k, v in sorted(dict2013.items(), key=lambda item: item[1],reverse=True)}
    result_areas = list(dict2013.keys())[0:10]  # top 10 LGA(Name)

    acc_year = ['2013','2014','2015','2016','2017','2018']
    result_values = {}  # top 10 LGA(Value)
    for i in acc_year:
        tmp_arr = []
        for j in result_areas:
            tmp_arr.append(yearAccOccur[i][j])
        result_values[i] = tmp_arr

    # calculating [No, Diff, Change] in each year, by comparing with 2013
    result_values = list(result_values.values())
    result_dct = {}
    for i, v in enumerate(result_values):
        if i + 1 < len(result_values):

            prev_yr = result_values[i]
            cur_yr = result_values[i + 1]
            diff = [abs(x1 - x2) for (x1, x2) in zip(prev_yr, cur_yr)]
            pct = [round((abs(x1 - x2) / x1 * 100), 2) for (x1, x2) in zip(prev_yr, cur_yr)]

            format_array = [["{:>6}".format(x1),"{:>6}".format(x2),"{:>8}%".format(x3)] for (x1, x2, x3) in zip(cur_yr, diff, pct)]
            result_dct[str(2014 + i)] = format_array

    # start tablize data
    result_areas.insert(0,"")  # format 1st column
    col2013 = result_values[0]
    col2013.insert(0,['No.'])  # format 2nd column

    for k,v in result_dct.items():  # format the rest of columns
        tmp_lst = v[:]
        header = ['No.','Diff.','Change.']
        header = ["{:>7}".format(x) for x in header]
        tmp_lst.insert(0,header)  # insert header
        result_dct[k] = tmp_lst

    report = pd.DataFrame({'LGA': result_areas,
                           '2013':col2013,
                           '2014':result_dct['2014'],
                           '2015':result_dct['2015'],
                           '2016':result_dct['2016'],
                           '2017':result_dct['2017'],
                           '2018':result_dct['2018']})

    print(report)  # print table as 4.2b required

    return None




'''4.1.3=================================================================='''

def sta_date_to_week(datalist):
    """this function is to transfer date to weekday 0-6 refers to Mon-Sun,
        return dictionary of each weekday and counts, e.g0{0:6, 1:8}"""
    weeks = []
    for d in datalist:
        # print(d)
        tmp = str(d).strip().replace(' ', '').replace('\n', '').replace('\r', '').replace('\r\n', '')
        week = datetime.strptime(tmp, "%d/%m/%Y").weekday()  # date to weekday
        weeks.append(week)

    statistic_result = Counter(weeks)  # count element in lists
    return statistic_result


def tast413():

    # data to plot
    src2018 = gpd.read_file(shp_path2018)
    src2013 = gpd.read_file(shp_path2013)
    tmp2013 = sta_date_to_week(src2013['ACCIDENT_D'])
    tmp2018 = sta_date_to_week(src2018['ACCIDENT_D'])
    acci_weeks2013 = []
    acci_weeks2018 = []
    for i in range(7):
        acci_weeks2013.append(tmp2013[i])
        acci_weeks2018.append(tmp2018[i])

    # create plot, as 4.1.3a required
    n_groups = 7
    fig, ax = mplt.subplots()
    index = np.arange(n_groups)
    bar_width = 0.35
    opacity = 0.8

    rects1 = mplt.bar(index, acci_weeks2013, bar_width, alpha=opacity, color='b', label='2013')
    rects2 = mplt.bar(index + bar_width, acci_weeks2018, bar_width, alpha=opacity, color='g', label='2018')

    mplt.xlabel('days in week')
    mplt.ylabel('accident occurrence')
    mplt.title('occurrence by week')
    mplt.xticks(index + bar_width, ('Mon', 'Tues', 'Wed', 'Thur', 'Fri', 'Sat', 'Sun'))
    mplt.legend()
    mplt.tight_layout()
    mplt.show()


    '''(b)'''
    # yearAccOccur = acci_by_year()
    # acc_year = ['2013','2014','2015','2016','2017','2018']
    # for y in acc_year:
    #     yearAccOccur[y] = sum(yearAccOccur[y].values())
    # print(yearAccOccur)




'''4.2========================================================================'''

def temp():
    '''read LGA2017  把一整塊大陸溶一起'''
    path = os.path.join(os.path.curdir, "data", "RegionsLGA_2017", "LGA_2017_VIC.shp")
    LGAdata = gpd.read_file(path)
    LGAdata = LGAdata.dropna()


    LGAdata = LGAdata[['STE_NAME16', 'geometry']]
    victoria = LGAdata.dissolve(by='STE_NAME16')

    # victoria.plot()
    # mplt.show()

    # LGAdata.to_file("testpackage.gpkg", layer='demoLGA', driver="GPKG")  # write to geopackage

'''4.2.1  Accident by LGA'''
def acc_by_lga():
    result = acci_by_year()  #statistic of 2013-2018 by all LGAs
    # aggregation by LGA
    lga_aggre = {}
    for lga in set(result['2018'])|set(result['2017'])|set(result['2016'])|set(result['2015'])|set(result['2014'])|set(result['2013']):
        lga_aggre[lga] = result['2018'].get(lga, 0)+\
                         result['2017'].get(lga, 0)+\
                         result['2016'].get(lga, 0)+\
                         result['2015'].get(lga, 0)+\
                         result['2014'].get(lga, 0)+\
                         result['2013'].get(lga, 0)
    #print(lga_aggre) # 88 LGAs in total
# acc_by_lga()



'''### 4.2.2  NEW LAYER AccidentLocations'''
def statistic_vc_type(srcdf):

    srcdf = srcdf[['HEAVYVEHIC', 'MOTORCYCLE', 'PASSENGERV', 'PUBLICVEHI']]
    resulttypelist = []
    for i in range(len(srcdf)):
        # print(srcfile.loc[[i]])  # dataframe type, per row
        val = srcdf.loc[i].values  # get values of each row ex [0,0,2,3]
        idx = srcdf.loc[i].index  # get index name of each row
        res = val.nonzero()  # important, this return only the index ***in integer of which element not zero
        t = str(list(idx[res]))
        resulttypelist.append(t)  # idx[res] is to get index name by index(integer)
    return resulttypelist


def task422():

    newlayerdf = pd.DataFrame(columns=['ACCIDENT_N','VEHICLE_T','DAY_OF_WEE','TOTAL_PERS','SEVERITY','geometry'])
    newlayerdf = gpd.GeoDataFrame(newlayerdf, geometry=newlayerdf.geometry, crs='EPSG:4283')  # convert to geodf so as to write to gpck

    for path in path_arr:
        srcfile = gpd.read_file(path)
        srcfile = srcfile.to_crs('EPSG:4283')
        data = statistic_vc_type(srcfile)  #s tatistic vehicle type first
        srcfile = srcfile.assign(VEHICLE_T=data)  # VEHICLE TYPE 待處理  因為有四種TYPE都是0 總count卻=1的是要寫哪種type
        masked_src = srcfile[srcfile.TOTAL_PERS>=3]  # important filter by value

        result = masked_src[['ACCIDENT_N','VEHICLE_T','DAY_OF_WEE','TOTAL_PERS','SEVERITY','geometry']]  # filtering wanted column
        newlayerdf = newlayerdf.append(result, ignore_index=True)

    newlayerdf.to_file("AccidentLocations.gpkg", layer='point', driver="GPKG")  # write to geopackage file, as 4.2.2required
    return newlayerdf



def task423():
    """task4.2.3  SA2 LAYER"""

    # first df, read LGA2017 and dissolving as a whole victoria boundary
    path1 = os.path.join(os.path.curdir, "data", "RegionsLGA_2017", "LGA_2017_VIC.shp")
    LGAdata = gpd.read_file(path1)[['STE_NAME16', 'geometry']].dropna()
    victoria = LGAdata.dissolve(by='STE_NAME16')  # dissolve by column ste_name and the same name

    # second df
    path2 = os.path.join(os.path.curdir, "data", "RegionsSA2_2016", "SA2_2016_AUST.shp")
    LGAaus = gpd.read_file(path2).dropna()

    # retrieve all the SA2name by the victoria boundary, reduce final calculation
    vicsa2 = gpd.overlay(LGAaus, victoria, how='intersection')

    sa2name = vicsa2[['SA2_NAME16', 'geometry']]
    acc_pts = task422()  # AccidentLocations layer from task422
    accpts_within_sa2name = gpd.sjoin(acc_pts, sa2name, how="left", op='intersects')

    print(accpts_within_sa2name.head())  # have a look at the top five rows of result



'''4.3==========================================================================='''

import re
from pysal.lib import weights
import seaborn as sns
from esda.moran import Moran, Moran_Local
from splot.esda import moran_scatterplot, lisa_cluster, plot_local_autocorrelation


def task43():

    # read LGA2017
    LGAdata = gpd.read_file(path_lga2017)
    LGAdata['ACCI_NUM_2013'] = 0  # create one more column with val 0
    acc_num_2013 = statistic_acc(shp_path2013)  # accident val by LGA in 2013

    for i, name in enumerate(LGAdata['LGA_NAME17']):
        for k, v in acc_num_2013.items():
            clean_name_13 = re.sub(r'[^\w\s]', '', k).replace(' ', '')
            clean_name_17 = re.sub(r'[^\w\s]', '', name).replace(' ', '')

            if clean_name_13.lower() in clean_name_17.lower():
                LGAdata.loc[i, 'ACCI_NUM_2013'] = v

    # print(LGAdata)

    f, ax = mplt.subplots(1, figsize=(9, 9))
    LGAdata.plot(column='ACCI_NUM_2013', scheme='Quantiles', legend=True, ax=ax)
    ax.set_axis_off()
    f.set_facecolor('0.75')
    # Title
    f.suptitle('Num of Accident happens', size=30)
    # Draw
    # mplt.show()


    LGAdata = LGAdata.dropna()  # important drop row with value none/na
    w = weights.Queen.from_dataframe(LGAdata, idVariable='LGA_CODE17')  # compute spatial weight matrix
    w.transform = 'R'  # Row standardize the matrix

    # calculate weight (spatial lag) and store into new column in LGAdata
    LGAdata['w_Acci_Num'] = weights.lag_spatial(w, LGAdata['ACCI_NUM_2013'])

    # Standardize/weight Standardize (spatial lag) and store into new column in LGAdata
    LGAdata['Acci_Num_std'] = (LGAdata['ACCI_NUM_2013'] - LGAdata['ACCI_NUM_2013'].mean()) / LGAdata['ACCI_NUM_2013'].std()
    LGAdata['w_Acci_Num_std'] = weights.lag_spatial(w, LGAdata['Acci_Num_std'])


    """Global Spatial autocorrelation"""
    fig, axs = mplt.subplots(1, 2, figsize=(15, 7))  # set figure

    # fig1 Moran scatterplot
    sns.regplot(x='Acci_Num_std', y='w_Acci_Num_std', data=LGAdata, ci=None, ax=axs[0])
    axs[0].axvline(0, c='k', alpha=0.5)
    axs[0].axhline(0, c='k', alpha=0.5)
    axs[0].set_title('Moran Scattorplot')

    # fig2 Moran scatterplot()
    moran = Moran(LGAdata['ACCI_NUM_2013'], w)  # Moran's I
    moran_scatterplot(moran, aspect_equal=True, ax=axs[1])

    mplt.show()


    """Local Spatial autocorrelation"""
    fig, axs = mplt.subplots(1, 2, figsize=(20, 7), gridspec_kw={'width_ratios': [1.5, 2]})  # set figure

    # fig1, scatter plot and four quadrants
    sns.regplot(x='Acci_Num_std', y='w_Acci_Num_std', data=LGAdata, ci=None, ax=axs[0])
    # Add vertical and horizontal lines
    axs[0].axvline(0, c='k', alpha=0.5)
    axs[0].axhline(0, c='k', alpha=0.5)
    axs[0].text(1.75, 0.5, "HH", fontsize=25)
    axs[0].text(1.75, -0.5, "HL", fontsize=25)
    axs[0].text(-0.75, 0.5, "LH", fontsize=25)
    axs[0].text(-0.75, -0.5, "LL", fontsize=25)
    # mplt.savefig('moran.png', dpi=300)

    # fig2, Local Indicators of Spatial Association(LISAs)
    lisa = Moran_Local(LGAdata['ACCI_NUM_2013'], w)
    LGAdata['significant'] = lisa.p_sim < 0.5  # threshold, Break observations into significant or not
    LGAdata['quadrant'] = lisa.q
    lisa_cluster(lisa, LGAdata, ax=axs[1])
    # mplt.savefig('LISAmap.png', dpi=300)

    mplt.show()

    return None




