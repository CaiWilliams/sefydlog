from smarts import * 
from gas_ppmv import *
import numpy as np
import copy
from PIL import Image
import datetime
import os
import matplotlib.image
import matplotlib.pyplot as plt
import cv2
import matplotlib.colors
import pandas as pd
import multiprocessing
import tqdm


EAC4 = pygrib.open('EAC4_2019_SeaLevel.grib')
EGG4 = pygrib.open('EGG4_2019_SeaLevel.grib')
A = atmosphere_analysis(EAC4,EGG4,2019,1,1,0,0,0)
A.get_gasses()
print(np.max(A.carbon_dioxide))
A.year = 2019
A.month = 12
A.day = 31
A.get_gasses()
print(np.max(A.carbon_dioxide))