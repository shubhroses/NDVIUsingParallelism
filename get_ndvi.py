"""
 MPI_INIT      		:		 Initiate an MPI computation.
 MPI_FINALIZE  		:		 Terminate a computation.
 MPI_COMM_SIZE		:		 Determine number of processes.
 MPI_COMM_RANK		:		 Determine my process identifier.
 MPI_SEND      		:		 Send a message.
 MPI_RECV      		:		 Receive a message.
"""
import time
from mpi4py import MPI
import numpy
numpy.seterr(divide='ignore', invalid='ignore')
import math
import rasterio
from rasterio import plot

comm = MPI.COMM_WORLD
rank = comm.Get_rank() # To determine current processor number

# Landsat 7
# image_height = 7251
# width = 8231
# red_band_path = './Landsat7/LE07_L1TP_042035_20220523_20220831_02_T1/LE07_L1TP_042035_20220523_20220831_02_T1_B3.TIF'
# nir_band_path = './Landsat7/LE07_L1TP_042035_20220523_20220831_02_T1/LE07_L1TP_042035_20220523_20220831_02_T1_B4.TIF'

# Landsat 8
# image_height = 7951
# width = 7831
# red_band_path = './Landsat8/LC08_L1TP_042035_20220427_20220503_02_T1/LC08_L1TP_042035_20220427_20220503_02_T1_B4.TIF'
# nir_band_path = './Landsat8/LC08_L1TP_042035_20220427_20220503_02_T1/LC08_L1TP_042035_20220427_20220503_02_T1_B5.TIF'

# Landsat 9
# image_height = 7941
# width = 7821
# red_band_path = './Landsat9/LC09_L1TP_042035_20220505_20220505_02_T1/LC09_L1TP_042035_20220505_20220505_02_T1_B4.TIF'
# nir_band_path = './Landsat9/LC09_L1TP_042035_20220505_20220505_02_T1/LC09_L1TP_042035_20220505_20220505_02_T1_B5.TIF'

# Presentation
startTime = time.time()

image_height = 1338
width = 2107
red_band_path = './presentation/Landsat8/LC08_L1TP_042035_20180603_20180615_01_T1_B4_clip.tif'
nir_band_path = './presentation/Landsat8/LC08_L1TP_042035_20180603_20180615_01_T1_B5_clip.tif'


def get_bands():
    # Return: numpy arrays of red and nir bands
    band3 = rasterio.open(red_band_path) # red
    band4 = rasterio.open(nir_band_path) # nir
    red = band3.read(1).astype('float64')
    nir = band4.read(1).astype('float64')
    return (red, nir)

def getNDVI(red, nir):
    # Input: Red and Nir bands
    # Result: numpy array with ndvi values 
    ndvi = numpy.where(
        (nir+red) == 0.,
        0,
        (nir-red)/(nir+red)
    )
    return ndvi

def get_subimage_height(r, h):
    # Input: Rank of processor and height of image
    # Result: The height of subimage assigned to that processor
    if r == comm.size-1:
        return h//comm.size
    else:
        return math.ceil(h/comm.size)

def get_received_image_height(r, h):
    # Input: Rank of processor and height of image
    # Result: The size of the image being sent to processor of rank r
    res = 0
    for i in range(r+1, comm.size):
        res += get_subimage_height(i, h)
    return res

sub_height = get_subimage_height(rank, image_height)

# Red and nir bands assigned to each processor
cur_red = None
cur_nir = None

if rank == 0:
    # Get red and nir bands for the image
    red, nir = get_bands()

    # Split up bands so each processor gets a slice of each
    red_split = numpy.array_split(red, comm.size)
    nir_split = numpy.array_split(nir, comm.size)

    # Send slices to respective processor
    for i in range(1, comm.size):
        comm.Send(red_split[i], dest=i, tag=77)
        comm.Send(nir_split[i], dest=i, tag=78)
    
    # Assign processor zero its slice 
    cur_red = red_split[0]
    cur_nir = nir_split[0]
elif rank > 0:
    # Each other processor receives its slice
    rec_red = numpy.ndarray(shape=(sub_height, width))
    rec_nir = numpy.ndarray(shape=(sub_height, width))
    comm.Recv(rec_red, source=0, tag=77)
    comm.Recv(rec_nir, source=0, tag=78)
    cur_red = rec_red
    cur_nir = rec_nir

cur_ndvi = getNDVI(cur_red, cur_nir)

plot.show(cur_ndvi, title = f"Processor ID: {rank}")

if rank == comm.size-1 and comm.size != 1:
    # Send calculated ndvi image to processors with rank - 1
    comm.Send(cur_ndvi, dest=rank-1, tag=89)
elif rank > 0 and rank < comm.size-1:
    # Receive image from processor rank + 1
    height_next_processor = get_received_image_height(rank, image_height)
    rec_ndvi = numpy.ndarray(shape=(height_next_processor, width))
    comm.Recv(rec_ndvi, source=rank+1, tag=89)

    # Concatenate own calculated ndvi image and send to processor rank -1
    combined = numpy.concatenate((cur_ndvi, rec_ndvi), axis=0)
    comm.Send(combined, dest=rank-1, tag=89)
elif rank == 0:
    if comm.size == 1:
        combined = cur_ndvi
    else:
        # Receive image from processor 1
        height_next_processor = get_received_image_height(rank, image_height)
        rec_ndvi = numpy.ndarray(shape=(height_next_processor, width))
        comm.Recv(rec_ndvi, source=1, tag=89)

        # Concatenate own calculated ndvi image
        combined = numpy.concatenate((cur_ndvi, rec_ndvi), axis=0)
    plot.show(combined)

if rank == 0:
    endTime = time.time()
    print(f"Time: {endTime - startTime} Number of Processors: {comm.size}")
