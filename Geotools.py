# %% Functions
# -*- encoding: utf-8 -*-
'''
@Time    :   2023/04/12 09:05:43
@Author  :   Qikang Zhao 
@Contact :   YC27963@umac.mo
@Description: 
'''
import warnings
warnings.filterwarnings('ignore')
import os
from pathlib import Path
import glob
from datetime import datetime, timedelta
import numpy as np
import pandas as pd
import xarray as xr
from osgeo import gdal,osr,ogr
import rasterio
from rasterio.merge import merge
import glob
import shutil
import geopandas as gpd
import salem

def check_dir(dir):
    # 查看文件夹是否存在, 如果不存在则创建一个
    if not os.path.exists(dir):
        os.mkdir(dir)
    return dir

def clear_dir(dir):
    # 清空一个文件夹内所有文件, 但不删除文件夹本身
    if os.path.exists(dir):
        shutil.rmtree(dir)
        os.mkdir(dir)
    else:
        os.mkdir(dir)
    return dir
        
def makenc(data, times=[0], lats=[0], lons=[0], varname='', savepath=''):
    # 保存nc文件
    times = np.array([times]).flatten()
    lats = np.array([lats]).flatten()
    lons = np.array([lons]).flatten()
    ds = xr.Dataset({varname: (['time','lat','lon'], data)},
                     coords={'time': times,'lat': lats,'lon': lons, })
    ds.to_netcdf(savepath, format='NETCDF4', 
                 encoding={varname: {'zlib': True, 'complevel': 4,'dtype':'float32'}})
    
def savetif(data, example='',savepath=''):
    '''Save predicted data'''
    extif = rasterio.open(example)
    height, width = extif.shape
    meta = {'driver': 'GTiff', 'dtype': 'float32', 'width': width, 'height': height,
        'transform':extif.transform,'count': 1, 'crs': extif.crs, 'compress': 'deflate'}
    if data.shape == (height, width):
        with rasterio.open(savepath,'w',**meta) as dst:
            dst.write(data,1)
    else:
        raise ValueError(f'Data shape {data.shape} does not match example shape {extif.shape}')

def blind_read(filepath):
    # 在无视变量名称的情况下盲读nc文件,返回：变量值，时间，纬度，经度
    ds = xr.open_dataset(filepath,engine='netcdf4')
    for var_name in ds.data_vars:
        dims = ds[var_name].dims
        if len(dims) == 3:#需要的变量是三维的
            return ds[var_name].values, ds.time.values, ds.lat.values, ds.lon.values

def mosaic_tif(file_dir, output_path):
    # 拼接分块tif
    file_paths = glob.glob(file_dir + "/*.tif")# 获取文件夹下所有栅格的路径
    datasets = []
    for path in file_paths:
        src = rasterio.open(path)
        datasets.append(src)
    mosaic_data, mosaic_transform = merge(datasets)
    mosaic_crs = src.crs
    profile = {
        'driver': 'GTiff',
        'height': mosaic_data.shape[1],
        'width': mosaic_data.shape[2],
        'count': 1,
        'dtype': mosaic_data.dtype,
        'crs': mosaic_crs,
        'transform': mosaic_transform}
    with rasterio.open(output_path, 'w', **profile) as dst:
        dst.write(mosaic_data)

        
def spatial_interpolate(path,savepath,lat_first,lat_res,lat_nums,lon_first,lon_res,lon_nums):
    '''
    # nc文件空间插值
    path:打开路径
    savepath:保存路径
    lat_first,lat_res,lat_nums:起始纬度,间隔(分辨率),纬度数量
    lon_first,lon_res,lon_nums:起始经度,间隔(分辨率),经度数量
    '''
    ds = xr.open_dataset(path,engine='netcdf4')
    new_lats = np.linspace(lat_first,lat_first+(lat_nums-1)*lat_res,lat_nums)
    new_lons = np.linspace(lon_first,lon_first+(lon_nums-1)*lon_res,lon_nums)
    ds_new = ds.interp(lat=new_lats,lon=new_lons)
    encoding={var:{'zlib':True,'complevel':5,'dtype':'float32'} for var in ds.data_vars}
    ds_new.to_netcdf(savepath,encoding=encoding)

def mask_nc(ncpath='', savepath='', maskpath=''):
    '''
    # 裁剪nc文件
    ncpath:打开路径
    savepath:保存路径
    maskpath:用于裁剪的shapefile, 使用WGS 1984地理坐标系
    '''
    mask_shp = gpd.read_file(maskpath)
    ds = xr.open_dataset(ncpath)
    data_masked = ds.salem.roi(shape=mask_shp)
    data_masked.to_netcdf(savepath, 
                          encoding={var:{'zlib':True,'complevel':5,'dtype':'float32'} for var in ds.data_vars})

def zonal_nc(varname = '', ncpath='', maskpath='', csvpath='', statistics = 'mean', zone_col_name = 'zone'):
    '''
    # 分区统计nc 或
    varname: nc文件变量名称
    ncpath: 打开nc路径,nc默认形状(time,lat,lon)
    maskpath: 用于裁剪的shapefile, 使用WGS 1984地理坐标系
    csvpath: 分区统计结束保存为csv
    statistics: 统计类型,可选mean/min/max,默认mean
    save: True或False,表示是否保存分区裁剪后的nc
    zone_col_name: maskpath的属性表中表示分区名称的列名
    '''
    ds = xr.open_dataset(ncpath)
    mask_shp = gpd.read_file(maskpath)
    zonal_data = np.zeros((mask_shp.shape[0],ds.time.shape[0]))
    for i, row in mask_shp.iterrows():
        zone_shape = row['geometry']  # 获取几何形状
        data_masked = ds.salem.roi(geometry = zone_shape)
        if statistics == 'mean':
            data_masked_series = data_masked.mean(dim=('lat','lon'))
        elif statistics == 'min':
            data_masked_series = data_masked.min(dim=('lat','lon'))
        elif statistics == 'max':
            data_masked_series = data_masked.max(dim=('lat','lon'))
        elif statistics == 'sum':
            data_masked_series = data_masked.sum(dim=('lat','lon'))
        zonal_data[i,:] = data_masked_series[varname].values
    zonal_ds =  pd.DataFrame(zonal_data,
                             index = mask_shp[zone_col_name], 
                             columns = ds.time.values)
    zonal_ds.to_csv(csvpath)

def tif2nc(inpath='', outpath='', time='1', varname='var',sig_num=2):
    if Path(outpath).exists():
        return
    data = gdal.Open(inpath)    # 读取tif
    im_height,im_width = data.RasterYSize,data.RasterXSize  # 获取高度，数组第一维，上下方向元素长度，代表纬度范围;# 获取宽度，数组第二维，左右方向元素长度，代表经度范围
    im_geotrans = data.GetGeoTransform()  # 获取仿射矩阵，含有 6 个元素的元组
    im_data = data.GetRasterBand(1).ReadAsArray(xoff=0, yoff=0, win_xsize=im_width, win_ysize=im_height)
    im_data = np.expand_dims(im_data, axis=0) # 增加时间维作为第一维
    data_attr = dict(standard_name=varname, uints="Defalt")
    # 根据im_geotrans得到图像的经纬度信息
    im_lon = [im_geotrans[0] + i * im_geotrans[1] for i in range(im_width)]
    lon_attr = dict(standard_name="lon", uints="degree")
    im_lat = [im_geotrans[3] + i * im_geotrans[5] for i in range(im_height)]
    lat_attr = dict(standard_name="lat", uints="degree")
    im_nc= xr.Dataset({
            varname:(['time','lat','lon'],im_data,data_attr)},
            coords={"time":time, "lat":(["lat"],im_lat,lat_attr), "lon":(["lon"],im_lon,lon_attr)})
    im_nc.to_netcdf(outpath, encoding={varname: {'zlib': True, 'complevel': 4,'dtype':'float32','least_significant_digit':sig_num}})

def nc2tifs(ncpath='',varname='var',prefix='var_',Output_folder=''):
    ds = xr.open_dataset(ncpath)
    times,lats,lons = ds.time.values,ds.lat.values, ds.lon.values
    data = ds[varname].values 
    LonMin,LatMax,LonMax,LatMin = [lons.min(),lats.max(),lons.max(),lats.min()]    #影像的左上角和右下角坐标
    #分辨率计算
    N_Lat,N_Lon = len(lats),len(lons)
    Lon_Res = (LonMax - LonMin) /(float(N_Lon)-1)
    Lat_Res = (LatMax - LatMin) / (float(N_Lat)-1)
    for i in (range(data.shape[0])): #时间数
        driver = gdal.GetDriverByName('GTiff')        #创建.tif文件
        outpath = os.path.join(Output_folder,prefix+times[i])
        out_tif = driver.Create(outpath,N_Lon,N_Lat,1,gdal.GDT_Float32)#长和宽,顺序不能反
        # 设置影像的显示范围
        #-Lat_Res一定要是-的
        geotransform = (LonMin,Lon_Res, 0, LatMax, 0, -Lat_Res)
        out_tif.SetGeoTransform(geotransform)
        #获取地理坐标系统信息，用于选取需要的地理坐标系统
        srs = osr.SpatialReference()
        srs.ImportFromEPSG(4326) # 定义输出的坐标系为"WGS 84"，AUTHORITY["EPSG","4326"]
        out_tif.SetProjection(srs.ExportToWkt()) # 给新建图层赋予投影信息
        #数据写出
        out_tif.GetRasterBand(1).WriteArray(data[i][::-1])
        # 将数据写入内存，此时没有写入硬盘 此处[::-1]用于图像的垂直镜像对称，避免图像颠倒
        out_tif.FlushCache() # 将数据写入硬盘
    del out_tif # 注意必须关闭tif文件