"""This is GEOM90042 Assignment3
Credit: Yuwei Tsai 1071545
"""

import rasterio as rio
from rasterio import plot as rplot
import numpy as np
from pyproj import Transformer
import matplotlib.pyplot as mplt
import matplotlib.patches as patches

# set the path to raster and check exist
import os
raster_path = os.path.join(os.path.curdir, "CLIP.tif")
os.path.exists(raster_path)


'''TASK1 STARTED-------------------------------------------------------------------------------------------------'''
def summary_dem(filename):
    """the function is to read dem file"""

    with rio.open(filename, 'r') as dem:
        band1_arr = dem.read(1)  # gets the band1 array read in memory

        bounds = dem.bounds  # get bounds data of the dem

        transformer = Transformer.from_crs(dem.crs, 'epsg:3111')
        min_x, min_y = transformer.transform(bounds.bottom, bounds.top)  # transform(lat, lon)
        max_x, max_y = transformer.transform(bounds.top, bounds.right)

    result_dic = {
        'name': dem.name,
        'coord_sys': dem.crs,

        'bounds_minX': min_x,
        'bounds_minLon': bounds.left,
        'bounds_maxX': max_x,
        'bounds_maxLon': bounds.right,
        'bounds_minY': min_y,
        'bounds_minLat': bounds.bottom,
        'bounds_maxY': max_y,
        'bounds_maxLat': bounds.top,

        'file_wdh': dem.width,
        'file_hgt': dem.height,
        'cell_size': dem.res[0],
        'nodata': dem.nodata,
        'min_val': band1_arr.min(),
        'max_val': band1_arr.max(),
    }
    dem.close()
    return result_dic


def display_summary(summary_dict):
    """the function is to display data as a pretty table"""

    data_header = ["Parameter", "Value"]
    data = [
        ["Filename", summary_dict['name']],
        ["Coordinate system", f"{summary_dict['coord_sys']} [EPSG]"],
        ["Min x, Min Lon", f"{summary_dict['bounds_minX']} [meters], {summary_dict['bounds_minLon']} [degrees]"],
        ["Max x, Max Lon", f"{summary_dict['bounds_maxX']} [meters], {summary_dict['bounds_maxLon']} [degrees]"],
        ["Min y, Min Lat", f"{summary_dict['bounds_minY']} [meters], {summary_dict['bounds_minLat']} [degrees]"],
        ["Max y, Max Lat", f"{summary_dict['bounds_maxY']} [meters], {summary_dict['bounds_maxLat']} [degrees]"],
        ["Width, Height, Cell size",
         f"{summary_dict['file_wdh']} [px], {summary_dict['file_hgt']} [px], {summary_dict['cell_size']} [px]"],
        ["NoData", summary_dict['nodata']],
        ["Min value, max value", f"{summary_dict['min_val']} [height], {summary_dict['max_val']} [height]"]
    ]

    # find the longest length of string in 1st_col and 2nd_col
    tmp_arr = np.array(data)
    mylen = np.vectorize(len)
    col1st_max, col2nd_max = mylen(tmp_arr).max(axis=0)

    # print in table format
    format_row = f"{{:<{col1st_max + 5}}}{{:<{col2nd_max}}}"
    print(format_row.format(*data_header))
    print("-" * (col1st_max + 5 + col2nd_max), "\n")
    for row in data:
        print(format_row.format(*row))

    return None


def output_geo_coor(filename):
    """This function is to output the coordinate of the highest value in array, and plot highlight"""

    with rio.open(filename, 'r') as src:
        src_arr = src.read(1)

        # fine the index of the max value in array
        index_row, index_col = np.where(src_arr == src_arr.max())
        # print(f'shape = {src.shape}, and index = {index_row}, {index_col}')

        mplt.imshow(src_arr)

        ax = mplt.gca()

        rec_width = 200
        rec_height = 200
        rect = patches.Rectangle((index_col - (rec_width / 2), index_row - (rec_height / 2)),
                                 rec_width,
                                 rec_height,
                                 linewidth=2,
                                 edgecolor='cyan',
                                 fill=False)
        ax.add_patch(rect)

    mplt.show()
    src.close()
    return src.xy(index_row, index_col)


'''TASK2 STARTED-------------------------------------------------------------------------------------------------'''
from rasterio.warp import calculate_default_transform, reproject, Resampling


def reproj_rsp_file(src_file_name, out_file_name, scale_factor):
    """The function is to reproject and resample the raster, and
    + Generate 2x and 4x cell-sized versions of the dataset
    + Output the histograms and tables
    as task 5.2.1 required.
    """

    dst_crs = 'EPSG:3111'
    pcent = 1 / scale_factor

    # Reopen the file
    vic_dem = rio.open(src_file_name)

    # creating the transform
    transform, width, height = calculate_default_transform(
        vic_dem.crs,
        dst_crs,
        vic_dem.width * pcent,
        vic_dem.height * pcent,
        *vic_dem.bounds,
    )

    # build the metadata
    kwargs = vic_dem.meta.copy()
    kwargs.update({
        'crs': dst_crs,
        'transform': transform,
        'width': width,
        'height': height

    })

    # generate output file, as required
    with rio.open(out_file_name, 'w', **kwargs) as res:
        for i in range(1, vic_dem.count + 1):
            reproject(
                source=rio.band(vic_dem, i),
                destination=rio.band(res, i),
                src_transform=vic_dem.transform,
                src_crs=vic_dem.crs,
                dst_transform=transform,
                dst_crs=dst_crs,
                dst_nodata=0,  # this is to prevent from NaN
                resampling=Resampling.bilinear)
    vic_dem.close()
    res.close()

    # show histogram, and print table, as required
    with rio.open(out_file_name, 'r') as data:
        rplot.show_hist(data,
                        bins=200,
                        histtype='stepfilled',
                        title='Histogram of the elevation values',
                        )

        dic = summary_dem(out_file_name)

        # transform back to 4326
        transformer = Transformer.from_crs(data.crs, 'epsg:4326')
        min_Lon, min_Lat = transformer.transform(data.bounds.bottom, data.bounds.top)  # transform(lat, lon)
        max_Lon, max_Lat = transformer.transform(data.bounds.top, data.bounds.right)
        dic['bounds_minLon'] = min_Lon
        dic['bounds_minLat'] = min_Lat
        dic['bounds_maxLon'] = max_Lon
        dic['bounds_maxLat'] = max_Lat

        display_summary(dic)
    data.close()

    return None


'''lets call the function, see the result of task5.2.1 required.(on Jupyter)'''
#src_file_name = 'CLIP.tif'
#out_file_name = f'reProj_rsp_2ndnd' + src_file_name
#reproj_rsp_file(src_file_name, out_file_name, 2)

# out_file_name = f'reProj_rsp_4' + src_file_name
# reproj_rsp_file(src_file_name, out_file_name, 4)

# dem = rio.open('reProj_rsp_2ndndCLIP.tif','r')
# rplot.show_hist(dem, bins=200, histtype='stepfilled')



# Start to calculate the Slope
def gaussianFilter(sizex, sizey=None, scale=0.333):
    """
    Generate and return a 2D Gaussian function
    of dimensions (sizex,sizey)
    If sizey is not set, it defaults to sizex
    A scale can be defined to widen the function (default = 0.333)
    """

    sizey = sizey or sizex
    x, y = np.mgrid[-sizex:sizex + 1, -sizey:sizey + 1]
    g = np.exp(-scale * (x ** 2 / float(sizex) + y ** 2 / float(sizey)))
    return g / g.sum()


def sec_fd(dem, cell_size):
    """This is the Second-order finite difference (2FD) method
        Base on convolution function, calculate the slope of a DEM
        """

    from scipy import signal
    f0 = gaussianFilter(3)
    I = signal.convolve(dem, f0, mode='valid')  # applies smooothing by gaussian filter
    f1 = np.array([[0, 0, 0], [1, 0, -1], [0, 0, 0]])  # 2FD filter
    f2 = f1.transpose()
    g1 = signal.convolve(I, f1, mode='valid') / cell_size*2
    g2 = signal.convolve(I, f2, mode='valid') / cell_size*2

    slope = np.arctan(np.sqrt(g1 ** 2 + g2 ** 2))

    return slope


def maxmax(dem_arr, cell_size):
    """This is the Maximum max method
        Base on window function, calculate the slope of a DEM
        """

    tmp_arr = np.full(dem_arr.shape, -1.0)  # array that full of -1

    tmp_arr[1:-1, 1:-1] = np.max([
        abs(dem_arr[1:-1, 1:-1] - dem_arr[:-2, :-2]) / np.sqrt(2) * cell_size,
        abs(dem_arr[1:-1, 1:-1] - dem_arr[1:-1, :-2]) / cell_size,
        abs(dem_arr[1:-1, 1:-1] - dem_arr[2:, :-2]) / np.sqrt(2) * cell_size,
        abs(dem_arr[1:-1, 1:-1] - dem_arr[:-2, 1:-1]) / cell_size,
        abs(dem_arr[1:-1, 1:-1] - dem_arr[2:, 1:-1]) / cell_size,
        abs(dem_arr[1:-1, 1:-1] - dem_arr[:-2, 2:]) / np.sqrt(2) * cell_size,
        abs(dem_arr[1:-1, 1:-1] - dem_arr[1:-1, 2:]) / cell_size,
        abs(dem_arr[1:-1, 1:-1] - dem_arr[2:, 2:] / np.sqrt(2) * cell_size)
    ], axis=0)

    slope = tmp_arr

    return slope


def compute_slope(filename):
    """this is the function which implements the 2 slope functions above as 5.2.2 required"""

    with rio.open(filename, 'r+') as src:
        data = src.read(1)

        # first slope function(2FD) and output
        slope_fd = sec_fd(data, src.res[0])

        # second slope function(max-max) and output
        slope_mm = maxmax(data, src.res[0])
    src.close()
    return slope_fd, slope_mm



def comparison(size2_filename, size4_filename):
    """Finally, do the comparison as 5.2.3 required, plot it"""

    # Compute slope
    slope_fd2x, slope_mm2x = compute_slope(size2_filename)
    slope_fd4x, slope_mm4x = compute_slope(size4_filename)
    print('Compute Slope Done!')

    # Important! If you want to compare, you need to normalize value, and flatten as well
    slope_fd2x, slope_mm2x = slope_fd2x.flatten(), slope_mm2x.flatten()
    slope_fd4x, slope_mm4x = slope_fd4x.flatten(), slope_mm4x.flatten()
    slope_fd2x, slope_mm2x = (slope_fd2x/slope_fd2x.max() *90), (slope_mm2x/slope_mm2x.max() *90)
    slope_fd4x, slope_mm4x = (slope_fd4x/slope_fd4x.max() *90), (slope_mm4x/slope_mm4x.max() *90)

    # Plot and setting
    figs, ([[x2_2FD_hist, x2_MM_hist, x2_FD_MM],
            [x4_2FD_hist, x4_MM_hist, x4_FD_MM]]) = mplt.subplots(2, 3, figsize=(21, 12))


    x2_2FD_hist.hist(slope_fd2x, bins=100, histtype='stepfilled', facecolor='r', alpha=0.3, label="2FD")
    x2_2FD_hist.set_title('cellsizeX2_2FD distribution')
    x2_2FD_hist.set_xlabel('value')
    x2_2FD_hist.set_ylabel('frequency')

    x2_MM_hist.hist(slope_mm2x, bins=100, histtype='stepfilled', facecolor='g', alpha=0.3, label="MM")
    x2_MM_hist.set_title('cellsizeX2_MaxMax distribution')
    x2_MM_hist.set_xlabel('value')
    x2_MM_hist.set_ylabel('frequency')

    x2_FD_MM.hist(slope_fd2x, bins=100, histtype='stepfilled', facecolor='r', alpha=0.3, label="2FD")
    x2_FD_MM.hist(slope_mm2x, bins=100, histtype='stepfilled', facecolor='g', alpha=0.3, label="MM")
    x2_FD_MM.set_title('cellsizeX2 FD&MM comparison')
    x2_FD_MM.set_xlabel('value')
    x2_FD_MM.set_ylabel('frequency')


    x4_2FD_hist.hist(slope_fd4x, bins=100, histtype='stepfilled', facecolor='r', alpha=0.3, label="2FD")
    x4_2FD_hist.set_title('cellsizeX4_2FD distribution')
    x4_2FD_hist.set_xlabel('value')
    x4_2FD_hist.set_ylabel('frequency')

    x4_MM_hist.hist(slope_mm4x, bins=100, histtype='stepfilled', facecolor='g', alpha=0.3, label="MM")
    x4_MM_hist.set_title('cellsizeX4_MaxMax distribution')
    x4_MM_hist.set_xlabel('value')
    x4_MM_hist.set_ylabel('frequency')

    x4_FD_MM.hist(slope_fd4x, bins=100, histtype='stepfilled', facecolor='r', alpha=0.3, label="2FD")
    x4_FD_MM.hist(slope_mm4x, bins=100, histtype='stepfilled', facecolor='g', alpha=0.3, label="MM")
    x2_FD_MM.set_title('cellsizeX2 FD&MM comparison')
    x4_FD_MM.set_xlabel('value')
    x4_FD_MM.set_ylabel('frequency')

    mplt.show()
    return None


'''lets see the result of comparison of 2xsize raster and 4xsize raster, as 5.2.3 required(on Jupyter).'''
# comparison('reProj_rsp_2CLIP.tif', 'reProj_rsp_4CLIP.tif')

