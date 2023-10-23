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
import rasterio
import rioxarray as rxr
import glob
import shutil
import geopandas as gpd
import salem
from rasterio.warp import calculate_default_transform, reproject, Resampling


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
        
def makenc(data=[], times=[], lats=[], lons=[], varname='', unit = '',savepath='',sig_num=2):
    # 保存nc文件
    times = np.array([times]).flatten()
    lats = np.array([lats]).flatten()
    lons = np.array([lons]).flatten()
    data_attr = dict(standard_name=varname, units=unit)
    ds = xr.Dataset({varname: (['time','lat','lon'], data,data_attr)},
                     coords={'time': times,'lat': lats,'lon': lons, })
    ds.to_netcdf(savepath, format='NETCDF4', 
                 encoding={var:{'zlib':True,'complevel':4,'dtype':'float32','least_significant_digit':sig_num} for var in ds.data_vars})
    
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
    mosaic_data, mosaic_transform = rasterio.merge(datasets)
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

def spatial_interpolate(inpath='',savepath='',lat_first=-90,lat_res=1,lat_nums=180,\
                        lon_first=-180,lon_res=1,lon_nums=360,\
                        method='linear',sig_num=2):
    '''
    # nc文件空间插值
    path:打开路径
    savepath:保存路径
    lat_first,lat_res,lat_nums:起始纬度,间隔(分辨率),纬度数量
    lon_first,lon_res,lon_nums:起始经度,间隔(分辨率),经度数量
    method: 插值方法{"linear", "nearest", "quadratic", "cubic"}, default: "linear"
    '''
    ds = xr.open_dataset(inpath,engine='netcdf4')
    new_lats = np.linspace(lat_first,lat_first+(lat_nums-1)*lat_res,lat_nums)
    new_lons = np.linspace(lon_first,lon_first+(lon_nums-1)*lon_res,lon_nums)
    ds_new = ds.interp(lat=new_lats,lon=new_lons, method=method)
    ds_new.to_netcdf(savepath,encoding={var:{'zlib':True,'complevel':4,'dtype':'float32','least_significant_digit':sig_num} for var in ds_new.data_vars})

def mask_data(inputpath='', save=False, savepath='', maskpath='',sig_num=2):
    '''
    # 裁剪nc文件
    ncpath:打开路径
    savepath:保存路径
    maskpath:用于裁剪的shapefile, 使用WGS 1984地理坐标系
    '''
    mask_shp = gpd.read_file(maskpath)
    input_type = Path(inputpath).suffix()
    if input_type=='.nc':
        ds = xr.open_dataset(inputpath)
    elif input_type=='.tif':
        ds = rxr.open_rasterio(inputpath)
    else:
        print(f'Unknown input file type: {input_type[1:]}')
    
    data_masked = ds.salem.roi(shape=mask_shp)
    if save:
        data_masked.to_netcdf(savepath, encoding={var:{'zlib':True,'complevel':4,'dtype':'float32','least_significant_digit':sig_num} for var in ds.data_vars})

def zonal_nc(varname='', ncpath='', maskpath='', \
            savetype='csv', outputpath='\.csv', \
            sel_time=False, timestamp=[],\
            statistics='mean', zone_col_name='zone'):
    '''
    # 分区统计nc 或
    varname: nc文件变量名称
    ncpath: 打开nc路径,nc默认形状(time,lat,lon)
    maskpath: 用于裁剪的shapefile, 使用WGS 1984地理坐标系
    
    savetype: 支持输出csv格式或者shp格式;
    outputpath: 分区统计结束保存路径
    
    sel_time:支持选择某一个时间节点进行区域统计
    timestamp: 时间戳,列表形式,支持选择多个时间点。如['1999','2000']
    
    statistics: 统计类型,可选mean/min/max,默认mean
    zone_col_name: maskpath的属性表中表示分区名称的列名
    '''
    ds = xr.open_dataset(ncpath)
    if sel_time:
        ds = ds.sel(time=timestamp)
    lon_interval, lat_interval = ds['lon'].diff(dim='lon').values[0],ds['lat'].diff(dim='lat').values[0]
    grid_area = lon_interval*lat_interval
    mask_shp = gpd.read_file(maskpath)
    zonal_data = np.zeros((mask_shp.shape[0],ds.time.shape[0]))
    for i, row in mask_shp.iterrows():
        zone_shape = row['geometry']
        area = zone_shape.area
        if area <= grid_area:
            center = zone_shape.centroid
            lon,lat = center.x,center.y
            lat_idx = np.abs(ds.lat - lat).argmin()
            lon_idx = np.abs(ds.lon - lon).argmin()
            data_masked_series = ds.isel(lon=lon_idx, lat=lat_idx)
        else:
            data_masked = ds.salem.roi(geometry=zone_shape)
            zonal_stat_function = getattr(data_masked, statistics) 
            data_masked_series = zonal_stat_function(dim=('lat','lon'))#e.g., statistics == 'mean'
            if np.isnan(data_masked_series).all() or np.sum(data_masked_series):
                data_masked = ds.salem.roi(geometry=zone_shape, all_touched=True)
                data_masked_series = zonal_stat_function(dim=('lat','lon'))
                zonal_stat_function = getattr(data_masked, statistics)
        zonal_data[i,:] = data_masked_series[varname].values
    if savetype=='shp':
        new_attributes = np.concatenate((mask_shp.values, zonal_data), axis=1)
        times = ['T'+str(y) for y in ds.time.values]
        new_columns = list(mask_shp.columns) + list(times)#32+?
        df = pd.DataFrame(new_attributes,columns=new_columns)
        for column in df.columns:
            try:
                df[column] = df[column].astype(np.float32)
            except:
                pass
        new_shp = gpd.GeoDataFrame(df, geometry=mask_shp.geometry)
        new_shp.to_file(outputpath, driver='ESRI Shapefile')
    elif savetype=='csv':
        zonal_ds =  pd.DataFrame(zonal_data,
                                index = mask_shp[zone_col_name], 
                                columns = ds.time.values)
        zonal_ds.to_csv(outputpath)

    
def nc2tifs(ncpath='', varname='var', prefix='var_', output_folder=''):
    ds = xr.open_dataset(ncpath)
    times, lats, lons = ds.time.values, ds.lat.values, ds.lon.values
    data = ds[varname].values
    lon_min, lat_max, lon_max, lat_min = [lons.min(), lats.max(), lons.max(), lats.min()]  # 影像的左上角和右下角坐标
    # 分辨率计算
    n_lat, n_lon = len(lats), len(lons)
    lon_res = (lon_max - lon_min) / (float(n_lon) - 1)
    lat_res = (lat_max - lat_min) / (float(n_lat) - 1)
    for i in range(data.shape[0]):  # 时间数
        outpath = os.path.join(output_folder, prefix + times[i] + '.tif')
        transform = rasterio.from_origin(lon_min, lat_max, lon_res, lat_res)
        with rasterio.open(outpath, 'w', driver='GTiff', height=n_lat, width=n_lon, count=1, dtype=data.dtype,
                           crs='EPSG:4326', transform=transform) as dst:
            dst.write(data[i][::-1], 1)  # [::-1]用于图像的垂直镜像对称，避免图像颠倒
            
def tiff2nc(inpath='', outpath='', varname = 'value', projection=False):
    if os.path.exists(outpath):
        return
    with rasterio.open(inpath) as src:
        im_data = src.read()  # 读取第一个波段的数据
        src_geotrans = src.transform  # 获取地理转换信息
        src_crs = src.crs # 获取坐标系
        src_width = src.width
        src_height = src.height
        src_bounds = src.bounds # 获取图像范围
    # 如果具有投影坐标系，则需要处理坐标转换
    if projection:
        # 获取源和目标坐标系信息
        dst_crs = rasterio.crs.CRS.from_epsg(4326)  # EPSG 4326 表示经纬度坐标系
        dst_transform, dst_width, dst_height = calculate_default_transform(
                src_crs,dst_crs, 
                src_width, src_height, 
                *src_bounds)
        profile = src.meta.copy()
        profile.update({'crs': dst_crs,'transform': dst_transform,'width': dst_width,'height': dst_height})
        del src
        # 重投影
        dst_array = np.empty((dst_height, dst_width), dtype=profile['dtype']) # 初始化输出图像数据
        reproject(source=im_data, src_crs=src_crs, src_transform=src_geotrans,#原始数据参数
                destination=dst_array, dst_transform=dst_transform, dst_crs=dst_crs,#目标参数
                resampling=Resampling.bilinear, num_threads=2)
        im_data = dst_array
        min_lon, _ = dst_transform*(0,0)
        _, min_lat = dst_transform*(dst_width, dst_height) 
        lon_res, lat_res = dst_transform.a, -dst_transform.e
        im_lon = [min_lon + i * lon_res for i in range(dst_width)]
        im_lat = [min_lat + i * lat_res  for i in range(dst_height)]
    else:
        min_lon, _ = src_geotrans*(0,0)
        _, min_lat = src_geotrans*(src_width, src_height) 
        im_lon = [min_lon + i * lon_res for i in range(src_width)]
        im_lat = [min_lat + i * lat_res  for i in range(src_height)]
    lon_attr = dict(standard_name="lon", uints="degrees")
    lat_attr = dict(standard_name="lat", uints="degrees")
    im_data = np.expand_dims(im_data, axis=0)  # 增加时间维
    im_data = np.where((im_data>1000)|(im_data<-1000), np.nan, im_data)
    data_attr = dict(standard_name=varname, uints=" ")
    dims = ['time', 'lat', 'lon']
    date = np.array([date]).flatten()
    im_lat = np.array([im_lat]).flatten()
    im_lon = np.array([im_lon]).flatten()
    im_nc = xr.Dataset({
        "LST": (dims, im_data, data_attr)},
        coords={"time": date,
                "lat": (["lat"], im_lat, lat_attr),
                "lon": (["lon"], im_lon, lon_attr)})
    del im_data, im_lat, im_lon
    new_lons = np.arange(-180,180,0.01)
    new_lats = np.arange(-90,90,0.01)
    im_nc.interp(lat=new_lats, lon=new_lons)
    im_nc.to_netcdf(outpath, encoding={varname: {'zlib': True, 'complevel': 5, 'dtype': 'float32', 'least_significant_digit': 0}})
    return im_nc, date

      