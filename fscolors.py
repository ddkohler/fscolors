"""
@author: Dan Kohler

"""

import os#, sys
#import time
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import matplotlib.colors as mplcolors
from mpl_toolkits.axes_grid1 import make_axes_locatable
from scipy.interpolate import griddata, interp1d
import matplotlib.gridspec as grd

matplotlib.rcParams['contour.negative_linestyle'] = 'solid'

#----------------------------STORE POSSIBLE PLOTTING AXES----------------------------

#functions to perform algebra on dat columns
def nm_to_wn(np_arr, invert=False):
    # invert is the same formula, so no change on inver argument
    return 10**7 / np_arr
def nm_to_eV(np_arr, invert=False):
    return 10**7 / (8067 * np_arr)
def position(z, d, invert=False):
    # convert delay in relative coord to absolute motor positions
    # d in fs, z in mm
    return z - d*3*1e-4 / 2.
def difference(a, b):
    return a - b
def add(a,b):
    return a + b

units = {
        'num':  (['num'], 0.0, None, 'acquisition number', None),
        'l1':   (['l1'], 1.0, 'nm', 'r$\mathrm{\lambda_1 (nm)}$', None),
        'l2':   (['l2'], 1.0, 'wn', r'$\mathrm{\lambda_2 (nm)}$', None),
        'l3':   (['l3'], 1.0, 'nm', r'$\mathrm{\lambda_3 (nm)}$', None),
        'lm':   (['lm'], 1.0, 'wn', r'$\mathrm{\lambda_m (nm)}$', None),
        'w1':   (['l1'], 5.0, 'wn', r'$\mathrm{\bar\nu_1 (cm^{-1})}$', nm_to_wn),
        'w2':   (['l2'], 5.0, 'nm', r'$\mathrm{\bar\nu_2 (cm^{-1})}$', nm_to_wn),
        'w3':   (['l3'], 5.0, 'wn', r'$\mathrm{\bar\nu_3 (cm^{-1})}$', nm_to_wn),
        'wm':   (['lm'], 1.0, 'nm', r'$\mathrm{\bar\nu_m (cm^{-1})}$', nm_to_wn),
        'w21':  (['w2', 'w1'], 5.0, 'wn', 
                 r'$\mathrm{\bar\nu_2 - \bar\nu_1 (cm^{-1})}$', difference),
        'E1':   (['l1'], 0.001, 'eV', r'$\mathrm{E_1 (eV)}$', nm_to_eV),
        'E2':   (['l2'], 0.001, 'eV', r'$\mathrm{E_2 (eV)}$', nm_to_eV),
        'E3':   (['l3'], 0.001, 'eV', r'$\mathrm{E_3 (eV)}$', nm_to_eV),
        'Em':   (['lm'], 0.001, 'eV', r'$\mathrm{E_m (eV)}$', nm_to_eV),
        'zref': (['zref'], 0.0005, 'mm', r'$\mathrm{z_{ref} (mm)}$', None),
        'z1':   (['z1', 'd1'], 0.00005, 'mm', r'$\mathrm{z_1 (mm)}$', position),
        'z2':   (['z2', 'd2'], 0.00005, 'mm', r'$\mathrm{z_2 (mm)}$', position),
        'dref': (['dref'], 25.0, 'fs', r'$\mathrm{d_{ref} (fs)}$', None),
        'd1':   (['d1'], 1.0, 'fs', r'$\mathrm{\tau_{2^{\prime}2} (fs)}$', -1),
        'ds1':  (['d1'], 1.0, 'fs', r'$d_1$', None),
        'd2':   (['d2'], 1.0, 'fs', r'$\mathrm{\tau_{12} (fs)}$', None),
        'd2real':(['d2', 'dref'], 2.0, 'fs', r'$\mathrm{\tau_{21} (fs)}$', difference),
        'ai0':  (['ai0'], 0.0, 'V', 'Signal 0', None),
        'ai1':  (['ai1'], 0.0, 'V', 'Signal 1', None),
        'ai2':  (['ai2'], 0.0, 'V', 'Signal 2', None),
        'ai3':  (['ai3'], 0.0, 'V', 'Signal 3', None),
        'ai4':  (['ai4'], 0.0, 'V', 'Signal 4', None)
}


# dictionary connects dat vars to plotting axes
# via scalars (e.g. femto --> pico) or functions (e.g. wavelength to eV) 
# functions can be of may dat column variables (e.g. convert delay to absolute position)
# default_txt[key] = ([datcol label arguments(s)], tolerance, units_str, name_str, func(datcol labels))
# functions will be applied only if Dat instance imports a file

#----------------------------ARRAY MANIPULATION CLASS----------------------------

class Dat:    
    """
        class creates a set of 3D coordinates (x,y,z) on a grid
            includes useful operations on such data sets
    """
    # let each instance have its own units instance
    units = units
    # the following dictionaries connects column names to locations for COLORS 
    #   scan
    # the value is the column from the dat file
    # v2 dictionary for dat convention implemented on 2014.03.25
    #v2_date = time.strptime('14 Mar 25', '%y %b %d')
    #v2_time = time.mktime(v2_date)    
    v2_time = 1395723600.0
    cols_v2 = {
        'num':  0,
        'l1':   1,
        'l2':   3,
        'l3':   5,
        'lm':   7,
        'zref': 11,
        'z1':   13,
        'z2':   15,
        'dref': 10,
        'd1':   12,
        'd2':   14,
        'ai0':  16,
        'ai1':  17,
        'ai2':  18,
        'ai3':  19,
        'ai4':  20
    }
    #v1_date = time.strptime('12 Oct 01', '%y %b %d')
    #v1_time = time.mktime(v1_date)
    v1_time = 1349067600.0
    cols_v1 = {
        'num':  0,
        'l1':   1,
        'l2':   3,
        'lm':   5,
        'd1':   6,
        't2p1': 6,
        'd2':   7,
        't21':  7,
        'ai0':  8,
        'ai1':  9,
        'ai2':  10,
        'ai3':  11
    }
    #the old column rules before Winter 2012 (when Skye changed the column assignments)
    cols_v0 = {
        'num':  0,
        'l1':   1,
        'l2':   3,
        'lm':   5,
        'd1':   6,
        't2p1': 6,
        'd2':   8,
        't21':  8,
        'ai0':  10,
        'ai1':  11,
        'ai2':  12,
        'ai3':  13
    }
    # colormaps

    blaise_cm = ['#0000FF',
                '#002AFF',
                '#0055FF',
                '#007FFF',
                '#00AAFF',
                '#00D4FF',
                '#00FFFF',
                '#FFFFFF',
                '#FFFF00',
                '#FFD400',
                '#FFAA00',
                '#FF7F00',
                '#FF5500',
                '#FF2A00',
                '#FF0000']

    signed_cm = ['#0000FF',# blue
                '#00BBFF', # blue-aqua
                '#00FFFF', # aqua
                '#FFFFFF', # white
                '#FFFF00', # yellow
                '#FFBB00', # orange
                '#FF0000'] # red   

    skye_amp = ['#000000', # white
                '#0000FF', # blue
                #'#00FFFF', # aqua
                '#00FF00', # lime
                '#FFFFFF', # black
                '#FFFF00', # yellow
                '#FF0000', # red
                '#881111'] # a dark red (no standard name)

    wrightcolors = ['#FFFFFF', # white
                '#0000FF', # blue
                '#00FFFF', # aqua
                '#00FF00', # lime
                '#FFFF00', # yellow
                '#FF0000', # red
                '#881111'] # a dark red (no standard name)
    # define colormaps
    wrightcm = mplcolors.LinearSegmentedColormap.from_list('wright',wrightcolors)
    mycm = wrightcm
    altcm = mplcolors.LinearSegmentedColormap.from_list('signed',signed_cm)
    altcm2 = mplcolors.LinearSegmentedColormap.from_list('signed2',blaise_cm)
    ampcm = mplcolors.LinearSegmentedColormap.from_list('skye',skye_amp)
    debug=False
    # font style attributes
    font_size = 14
    font_family = 'sans-serif'
    # plot windowing--use to concatenate image
    limits = False
    xlim = [-150,150]
    ylim = [-150,150]
    # attribute used to set z axis scale
    zmin,zmax = 0,1
    # attribute used to indicate the signal zero voltage
    # initial value is znull=zmin
    znull = 0.
    has_data=False
    # attributes of contour lines
    contour_n = 9
    contour_kwargs={'colors':'k',
                    'linewidths':2}
    # attributes of sideplots
    side_plot_proj_kwargs={'linewidth':2}
    side_plot_proj_linetype = 'b'
    side_plot_else_kwargs={'linewidth':2}
    side_plot_else_linetype = 'r'

    def __init__(self, scantype='2d', zvar='ai0', user_created=True, 
                 grid_factor=1, **kwargs):
        #filepath=None, 
        #xvar=None, yvar=None, 
        #cols=None,
        #grid_factor=grid_factor,
        #color_offset=None, 
        #font_size=None, font_family=None,
        #colortune=False, znull=None, 
        # can manually import gridded data
        #zi=None, xi=None, yi=None):
        filepath = kwargs.get('filepath')
        isfile=False
        if isinstance(filepath, str):
            if os.path.isfile(filepath):
                isfile=True
        zi = kwargs.get('zi')
        znull = kwargs.get('znull')
        xvar = kwargs.get('xvar')
        yvar = kwargs.get('yvar')
        cols = kwargs.get('cols')
        if cols == 'v2':
            self.datCols = Dat.cols_v2
        elif cols == 'v1':
            self.datCols = Dat.cols_v1
        elif cols == 'v0':
            self.datCols = Dat.cols_v0
        elif isfile:
            # guess based on when the file was made
            file_date = os.path.getctime(filepath)
            if file_date > Dat.v2_time:
                cols = 'v2'
                self.datCols = Dat.cols_v2
            elif file_date > Dat.v1_time:
                cols = 'v1'
                self.datCols = Dat.cols_v1
            else:
                # file is older than all other dat versions
                cols = 'v0'
                self.datCols = Dat.cols_v0
        else:
            # guess the most likely assignment:  v2
            self.datCols = Dat.cols_v2
        self.cols=cols
        # give each Dat instance its own copy of units (so we can autochange 
        # the scale)
        self.units = Dat.units.copy()
        # check that plotting axes are valid
        try:
            scantype = str.lower(scantype)
        except TypeError:
            pass
        if scantype == '2d':
            # define variables and make sure they are valid
            if xvar in self.units.keys():
                self.xvar = xvar
            else:
                print 'xvar {0} is not valid.  aborting import'.format(xvar)
                return
            if yvar in self.units.keys():
                self.yvar = yvar
            else:
                print 'yvar {0} is not valid.  aborting import'.format(yvar)
                return
            self.zvar = zvar
        elif scantype == '1d':
            # define variables and make sure they are valid
            if xvar in self.units.keys():
                self.xvar = xvar
            else:
                print 'xvar {0} is not valid.  aborting import'.format(xvar)
                return
            self.zvar = zvar
        elif scantype == 'norm': # file will not be explicitly plotted--need only self.data
            pass
        else:
            print 'no treatment known for scan type', scantype
        # if all the variable settings are okay, now import data 
        # either by file or by array
        if isfile:
            # check again to make sure the file still exists
            if os.path.isfile(filepath):
                self.has_data=True
                print 'found the file!'
            else:
                self.has_data = False
                print 'filepath',filepath,'does not yield a file'
                return        
            self.filepath, self.filename, self.file_suffix = filename_parse(filepath)
            # now import file as a local var
            rawDat = np.genfromtxt(filepath,dtype=np.float) 
            # define arrray used for transposing dat file so cols are first index
            self.grid_factor=grid_factor
            # allow data to be masked
            self.data = rawDat.T
            self.smoothing = False
            self.z_scale = 'linear'
            self.constants = []
            # define constants (anything that is not x,y or z) and extract values
            var_cols = [self.datCols.get(xvar), 
                        self.datCols.get(yvar),
                        self.datCols.get('ai0'),
                        self.datCols.get('ai1'),
                        self.datCols.get('ai2'),
                        self.datCols.get('ai3'),
                        self.datCols.get('ai4')
                        ]
            # now operate based on scantype
            if scantype == '2d':
                for key in self.datCols.keys():
                    col = self.datCols[key]
                    if col not in var_cols:
                        self.constants.append([key,self.data[col]])
                # shift assorted default values
                font_size = kwargs.get('font_size')
                font_family = kwargs.get('font_family')
                if font_size:
                    self.font_size = font_size
                if font_family:
                    self.font_family = font_family
                print 'file has imported'
                # column assignments no longer make sense if axes are abstracted
                #self.xcol = self.datCols[self.xvar]
                #self.ycol = self.datCols[self.yvar]
                #self.zcol = self.datCols[self.zvar]
    
                self.zvars = {
                        'ai0' : None,
                        'ai1' : None,
                        'ai2' : None,
                        'ai3' : None
                }
                # add another ai if the newest dat format
                if self.cols in ['v2']:
                    self.zvars['ai4'] = None
                self.xy = np.zeros((2,self.data.shape[-1]))
                # extract from the dat the appropriate x,y,z axes to plot
                # make data_x, data_y, columns to create variables?
                # it makes sense:  don't overwrite the dat array
                self.zcol = self.datCols[self.zvar]
                self.xy[0] = self._get_axes(self.xvar)
                self.xy[1] = self._get_axes(self.yvar)
                # if colortune, subtract off the central frequency from the mono axis
                # this gives units of displacement from the set point
                if kwargs.get('colortune'):
                    # figure out how to convert the mono axis
                    if self.xvar in ['lm', 'wm'] and self.yvar in ['l1', 'l2', 'w1', 'w2']:
                        if (self.xvar in ['lm'] and self.yvar in ['l1','l2']) or (
                            self.xvar in ['wm'] and self.yvar in ['w1','w2']):
                            self.xy[0] = self.xy[0] - self.xy[1]
                        else:
                            self.xy[0] = self.xy[0] - 10**7 / self.xy[1]
                    elif self.yvar in ['lm', 'wm'] and self.xvar in ['l1', 'l2', 'w1', 'w2']:
                        if (self.yvar in ['lm'] and self.xvar in ['l1','l2']) or (
                            self.yvar in ['wm'] and self.xvar in ['w1','w2']):
                            self.xy[1] = self.xy[1] - self.xy[0]
                        else:
                            self.xy[1] = self.xy[1] - 10**7 / self.xy[0]
                if user_created:
                    # create xy grid (vars self.xi, self.yi) and interpolate z values to grid
                    # new:  grid for all ai channels
                    if znull is not None:
                        self.znull = znull
                        self._gengrid()
                    else:
                        # if no znull, we probably want zmin to be our zero voltage
                        self.znull = self.data[self.zcol].min()
                        self._gengrid()
                        self.znull = self.zi.min()                    
                    # store min and max independent of data so different scaling can be applied
                    self.zmax = self.zi.max()
                    self.zmin = self.zi.min()
                else:
                    pass
            elif scantype == '1d':
                print self.xvar, self.zvar
                self.xi = self._get_axes(self.xvar)
                self.zi = self._get_axes(self.zvar)
            # check if a grid is being manually imported
            # don't know why i need this if is_file==True...
            elif isinstance(zi, np.ndarray):
                self.zi = zi
                self.zvar = zvar
                self.zmin = self.zi.min()
                self.zmax = self.zi.max()
                xi = kwargs.get('xi')
                if xi is None:
                    self.xi = np.array(range(len(zi.shape[1])))
                else:
                    self.xi = xi
                yi = kwargs.get('yi')
                if yi is None:
                    self.yi = np.array(range(len(zi.shape[0])))
                else:
                    self.yi = yi
                self.grid_factor=2
        # check if a grid is being manually imported
        elif isinstance(zi,np.ndarray):
            self.zi = zi
            self.xi = kwargs.get('xi')
            self.yi = kwargs.get('yi')
            if znull is None:
                # rewrite as an initial guess
                self.znull = self.zi.min()
            if kwargs.get('zmin') is None:
                self.zmin = self.zi.min()
            if kwargs.get('zmax') is None:
                self.zmax = self.zi.max()
            self.xvar = kwargs.get('xvar')
            self.yvar = kwargs.get('yvar')
            self.grid_factor=1
            self.zfit()
        else:
            print 'no data was imported'

    def _3PEPS_trace(self, T, w2w2p_pump = True):
        """
            Must accept 2d delay scan type
            Assumes typical input dimensions of tau_21 and tau_2p1
            Returns the coherence trace for a specified population time, T
        """
        tau_out = []
        if self.xvar == 'd1' or self.xvar == 't2p1':
            #define tolerances for delay value equivalence
            d1tol = self.units[self.xvar][1]
            d1_col = self.xcol
            d2tol = self.units[self.yvar][1]
            d2_col = self.ycol
        else:
            d1tol = self.units[self.yvar][1]
            d1_col = self.ycol
            d2tol = self.units[self.xvar][1]
            d2_col = self.xcol

        if w2w2p_pump:
            #flip sign (ds = -T)
            ds=-T
            #find values such that d2 = ds
            for i in range(len(self.data[0])):
                #find the horizontal part of the data we want
                if (np.abs(self.data[d2_col][i] - ds) <= d2tol) and (self.data[d1_col][i] - ds) <= d1tol:
                    #2p comes first (non-rephasing)
                    #print 'd1,d2 = %s, %s' % (self.data[d1_col][i],self.data[d2_col][i])
                    tau_out.append([
                        self.data[d1_col][i]-ds,
                        self.data[self.zcol][i]])
                
                elif np.abs(self.data[d1_col][i] - ds) <= d1tol and self.data[d2_col][i] - ds <= d2tol:
                    #2 comes first  (rephasing)
                    #print 'd1,d2 = %s, %s' % (self.data[d1_col][i],self.data[d2_col][i])
                    tau_out.append([
                        -(self.data[d2_col][i]-ds),
                        self.data[self.zcol][i]])
        else:
            #pump is w1w2
            #find values such that d2 = ds
            for i in range(len(self.data[0])):
                #find the slice across d2 we want (T=d1 and d2 <= 0)
                if (np.abs(self.data[d1_col][i] - T) <= d1tol) and self.data[d2_col][i] <= d2tol:
                    #2 comes first (rephasing)
                    tau_out.append([
                        -self.data[d2_col][i],
                        self.data[self.zcol][i]])
                #find the antidiagonal slice we want (d1-d2 = T and d2 >= 0)
                elif np.abs(self.data[d1_col][i] - self.data[d2_col][i] - T) <= d1tol and self.data[d2_col][i] >= d2tol:
                    #1 comes first  (non-rephasing)
                    tau_out.append([
                        -self.data[d2_col][i],
                        self.data[self.zcol][i]])
        #order the list
        tau_out.sort()
        tau_out = np.array(zip(*tau_out))
        return np.array(tau_out)

    def center(self, axis=None, center=None):
        if center == 'max':
            print 'listing center as the point of maximum value'
            if axis == 0 or axis in ['x', self.xvar]:
                index = self.zi.argmax(axis=0)
                set_var = self.xi
                max_var = self.yi
                out = np.zeros(self.xi.shape)
            elif axis == 1 or axis in ['y', self.yvar]:
                index = self.zi.argmax(axis=1)
                set_var = self.yi
                max_var = self.xi
                out = np.zeros(self.yi.shape)
            else:
                print 'Input error:  axis not identified'
                return
            for i in range(len(set_var)):
                out[i] = max_var[index[i]]
        else:
            # find center by average value
            out = self.exp_value(axis=axis, moment=1)
        return out
                
    def colorbar(self):
        """
            adds colorbar to the contour plot figure
            only after all contour plot embellishments have been performed
        """
        if self.s1:
            ax_cb = plt.subplot(self.gs[1])
        else:
            print 'must create plot before adding colorbar'
            return
        if self.alt_zi == 'int':
            ticks = np.linspace(-1,1,21)
            # find the intersection of the range of data displayed and ticks
            ticks = [ticki for ticki in ticks if ticki >= 
                min(self.zi_norm.min(), self.znull) and 
                ticki <= max(self.znull, self.zi_norm.max())]
            self.p1.colorbar(self.cax, ticks=ticks, cax=ax_cb)
        elif self.alt_zi == '+':
            ticks = np.linspace(self.znull,self.zmax,11)
            # find the intersection of the range of data displayed and ticks
            self.p1.colorbar(self.cax, ticks=ticks, cax=ax_cb)
        elif self.alt_zi == '-':
            ticks = np.linspace(self.zmin,self.znull,11)
            # find the intersection of the range of data displayed and ticks
            self.p1.colorbar(self.cax, ticks=ticks, cax=ax_cb)
        elif self.alt_zi == 'amp':
            ticks = np.linspace(0,1,11)
            ampmin = self.floor
            ampmax = self.ceiling
            ticks = np.linspace(ampmin,ampmax,num=11)
            # determine how much precision is necessary in the ticks:
            self.p1.colorbar(self.cax, ticks=ticks, cax=ax_cb)
        elif self.alt_zi == 'log':
            # the colorbar range
            # not sure whether to define the range using masked array or
            # full array
            #logmin = np.log10(masked.min() / (self.zmax - masked.min()))
            logmin = self.floor
            logmax = self.ceiling
            ticks = np.linspace(logmin,logmax,num=11)
            # determine how much precision is necessary in the ticks:
            decimals = int(np.floor(-np.log10(np.abs(
                ticks[-1]-ticks[0])))) + 2
            ticklabels = np.around(ticks,decimals)
            self.p1.colorbar(self.cax, ticks=ticks, cax=ax_cb).ax.set_yticklabels(ticklabels)
            """
            decimals = int(np.floor(-np.log10(np.abs(
                ticks[-1]-ticks[0])))) + 2
            ticklabels = np.around(ticks,decimals)
            self.p1.colorbar(self.cax, ticks=ticks, cax=ax_cb).ax.set_yticklabels(ticklabels)
            """
        elif self.alt_zi in [None, 'raw']: # raw colorbar
            ticks = np.linspace(min([self.znull, self.zmin]),
                                max(self.znull, self.zmax),num=11)
            decimals = int(np.floor(-np.log10(np.abs(
                ticks[-1]-ticks[0])))) + 2
            ticklabels = np.around(ticks,decimals)
            self.p1.colorbar(self.cax, ticks=ticks, cax=ax_cb).ax.set_yticklabels(ticklabels)
            #self.p1.colorbar(self.cax, ticks=ticks, cax=ax_cb)
        else: #could not determine colorbar type
            print 'color scale used not recognized:  cannot produce colorbar'

    def Dat(self):
        """
            output a copy of this Dat instance
        """
        out = Dat(zi=self.zi.copy(), xi=self.xi.copy(), yi=self.yi.copy(), 
                  xvar=self.xvar, yvar=self.yvar)
        return out
        
    def _diag(self, offset=0.0, use_griddata=False):
        """
            returns an array of z-axis points in the interpolated array that satisfy x=y
        """
        #check that x and y both are the same domain (i.e. 2dfreq or 2d delay)
        out=[]
        delays = ['d1','d2','t2p1','t21']
        freq = ['w1','w2']
        wavelength = ['l1','l2']
        if (self.xvar in delays and self.yvar in delays) or (self.xvar in freq and self.yvar in freq) or (self.xvar in wavelength and self.yvar in wavelength):
            if use_griddata:
                # alternate version:  use griddata
                #min_diag = max(min(self.xi),min(self.yi))
                #max_diag = min(max(self.xi),max(self.yi))
                # make grid values
                pass
            else:
                #initialize the closest we get with a random cloeseness number
                closest=np.abs(self.xi[0]-self.yi[0])
                #find the x and y coordinates that agree to within tolerance
                for i in range(len(self.xi)):
                    for j in range(len(self.yi)):
                        difference = np.abs(self.xi[i] - self.yi[j])
                        if difference <= self.datCols[self.xvar][0]:
                            out.append([
                                (self.xi[i]+self.yi[j])/2,
                                self.zi[j][i]])
                        else:
                            closest=min([closest,difference])
            #check if we have any values that fit
            if len(out) == 0:
                print 'no x and y values were closer than {0}.  Try increasing grid_factor'.format(closest)
            else:
                out.sort()
                out = np.array(zip(*out))
                return np.array(out)
        else:
            print 'cannot give diagonal if x and y units are not the same'
            print 'x axis:', self.xvar
            print 'y axis:', self.yvar
            return
        
    def difference2d(self):
        """
            Take the registered plot and import one to take the difference.
            Difference will be plotted as ref - imported (so biggest differences are red)
        """
        print 'Specify the requested file to compare with ref'
        imported = Dat(xvar=self.xvar, yvar=self.yvar, user_created=False, grid_factor=self.grid_factor)
        #create zi grid using ref grid values
        imported._gengrid(xlis=self.xi, ylis=self.yi)
        #imported and ref should have same zi grid size now--subtract and plot!
        #normalize both grids first
        zrefmax = self.zi.max()
        zrefmin = self.zi.min()
        zimpmax = imported.zi.max()
        zimpmin = imported.zi.min()
        
        self.zi = (self.zi - zrefmin) / (zrefmax - zrefmin)
        imported.zi = (imported.zi - zimpmin) / (zimpmax - zimpmin)

        diffzi =  imported.zi - self.zi
        self.plot2d(alt_zi=diffzi, scantype='Difference')

    def dOD(self, zvar='ai4'):
        """
        for differential scans:  convert zi signal from dT to dOD
        if Wigners, we average the entire delay value (assume constant power)
        # if we have a uncertainty level for our data, we will propagate a new 
        error matrix that can be used to mask data that is "blown up" by the 
        operation
        
        note that the below assumes TA data collected with boxcar i.e.
        sig = 1/2 dT
        ref = T + 1/2 dT
        """
        time = ['fs', 'ps', 'ns']
        xunit = self.units[self.xvar][2]
        yunit = self.units[self.yvar][2]
        if xunit in time and yunit not in time:
            T = self.zvars[zvar].sum(axis=1) / self.xi.shape
            dT = 2 * self.zi
            self.zi = -np.log10((T[:,None] + dT) / T[:,None])
        elif xunit not in time and yunit in time:
            T = self.zvars[zvar].sum(axis=0) / self.yi.shape
            dT = 2 * self.zi
            self.zi = -np.log10((T[None,:] + dT) / T[None,:])
        else:
            T = self.zvars[zvar]
            dT = 2 * self.zi
            self.zi = -np.log10((T + dT) / T)
        self.znull = 0.
        self.zmin = self.zi.min()
        self.zmax = self.zi.max()

    def export(self, fname=None):
        """
            export as xyz table
        """
    def export_dat(self, fname=None, cols='v2'):
        """
            generate a dat file using the current zi grid
            cols determines teh output format
            currently ignores constants of the scan
        """
        # will only work if mapping back to dat is easy
        out_x = self.xi
        out_y = self.yi
        # convert back to default dat units
        out_x = self._inv_get_axes(self.xvar)
        out_y = self._inv_get_axes(self.yvar)
        if cols=='v2':
            out = np.zeros((self.zi.size, 27))
            cols = Dat.cols_v2
        # index column is easy:
        print out.shape, self.zi.size
        out[:,0] = np.arange(out.shape[0])
        out_x = np.tile(out_x, self.zi.shape[0])
        out_y = np.tile(out_y[:,None], (1, self.zi.shape[1])).ravel()

        for zkey in self.zvars.keys():
            if zkey == self.zvar:
                # in case zi is altered
                out[:,cols[zkey]] = self.zi.ravel()
            else:
                out[:,cols[zkey]] = self.zvars[zkey].ravel()
        # can only do this if inversion is 1:1 mapping
        if len(units[self.xvar][0]) == 1:
            xcol = cols[units[self.xvar][0][0]]
        else:
            print 'cannot source correct column to place xvar'
        print self.zi.shape[0], out_x.shape, out_y.shape
        out[:,xcol] = out_x
        if len(units[self.yvar][0]) == 1:
            ycol = cols[units[self.yvar][0][0]]
        else:
            print 'cannot source correct column to place yvar'
            ycol
        out[:,ycol] = out_y
        
        if fname is None:
            fname = self.filename
            filepath = self.filepath
            file_suffix = 'dat'
        else:
            filepath, fname, file_suffix = filename_parse(fname)
            if not file_suffix:
                file_suffix = 'dat' 

        if filepath:
            fname = filepath + '\\' + fname
        fname = find_name(fname, file_suffix)
        fname += '.' + file_suffix
        np.savetxt(fname, out, fmt='%10.5F', delimiter='\t')
        print 'exported dat saved as {0}'.format(fname)

    def exp_value(self, axis=None, moment=1, norm=True, noise_filter=None):
        """
            returns the weighted average for fixed points along axis
            specify the axis you want to have exp values for (x or y)
            good for poor-man's 3peps, among other things
            moment argument can be any integer; meaningful ones are:
                0 (area, set norm False)
                1 (average, mu) or 
                2 (variance, or std**2)
            noise filter, a number between 0 and 1, specifies a cutoff for 
                values to consider in calculation.  zi values less than the 
                cutoff (on a normalized scale) will be ignored
            
        """
        if axis == 0 or axis in ['x', self.xvar]:
            # an output for every x var
            zi = self.zi.copy()
            int_var = self.yi
            out = np.zeros(self.xi.shape)
        elif axis == 1 or axis in ['y', self.yvar]:
            # an output for every y var
            zi = self.zi.T.copy()
            int_var = self.xi
            out = np.zeros(self.yi.shape)
        else:
            print 'Input error:  axis not identified'
            return
        if not isinstance(moment, int):
            print 'moment must be an integer.  recieved {0}'.format(moment)
            return
        for i in range(out.shape[0]):
            # ignoring znull for this calculation, and offseting my slice by min
            zi_min = zi[:,i].min()
            #zi_max = zi[:,i].max()
            temp_zi = zi[:,i] - zi_min
            if noise_filter is not None:
                cutoff = noise_filter * (temp_zi.max() - zi_min)
                temp_zi[temp_zi < cutoff] = 0
            #calculate the normalized moment
            if norm == True:
                out[i] = np.dot(temp_zi,int_var**moment) / temp_zi.sum()#*np.abs(int_var[1]-int_var[0]) 
            else:
                out[i] = np.dot(temp_zi,int_var**moment)
        return out

    def fit_gauss(self, axis=None):
        """
            least squares optimization of traces
            intial params p0 guessed by moments expansion
        """
        if axis == 0 or axis in ['x', self.xvar]:
            # an output for every x var
            zi = self.zi.copy()
            var = self.yi
            #out = np.zeros((len(self.xi), 3))
        elif axis == 1 or axis in ['y', self.yvar]:
            # an output for every y var
            zi = self.zi.T.copy()
            var = self.xi
            #out = np.zeros((len(self.yi), 3))

        # organize the list of initial params by calculating moments
        m0 = self.exp_value(axis=axis, moment=0, norm=False)
        m1 = self.exp_value(axis=axis, moment=1, noise_filter=0.1)
        m2 = self.exp_value(axis=axis, moment=2, noise_filter=0.1)        

        mu_0 = m1
        s0 = np.sqrt(np.abs(m2 - mu_0**2))
        A0 = m0 / (s0 * np.sqrt(2*np.pi))
        offset = np.zeros(m0.shape)
        
        print mu_0

        p0 = np.array([A0, mu_0, s0, offset])
        out = p0.copy()
        from scipy.optimize import leastsq
        for i in range(out.shape[1]):
            #print leastsq(gauss_residuals, p0[:,i], args=(zi[:,i], var))
            try:
                out[:,i] = leastsq(gauss_residuals, p0[:,i], args=(zi[:,i]-self.znull, var))[0]
            except:
                print 'least squares failed on {0}:  initial guesses will be used instead'.format(i)
                out[:,i] = p0[:,i]
            #print out[:,i] - p0[:,i]
        out[2] = np.abs(out[2])
        return out
        
    def _inv_get_axes(self, key):
        if key == self.xvar:
            args = self.xi
        elif key == self.yvar:
            args = self.yi
        else:
            raise TypeError()
        method = self.units[key][4]
        if method is None:
            return args
        elif isinstance(method, int) or isinstance(method, float):
            return method * args
        elif isinstance(method, object):
            return method(args, invert=True)
        else:
            print 'could not calculate axis for key {0}'.format(key)

    def _get_axes(self, key):
        """
            iterate the conversion until the conversion of interest is applied to 
                all dependencies
            returns a calculated column of interest (1xN, N is number of rows in data)
        """
        method = self.units[key][4]
        args_kw = self.units[key][0]
        al = []
        # make sure all arguments are calculated before we calculate the axis
        for i in range(len(args_kw)):
            a = args_kw[i]
            try:
                col = self.datCols[a]
            except KeyError:
                # if it's not a member of the raw dat, then we have to calculate it
                al.append(self._get_axes(a))
            else:
                # if it's a member of raw dat, give it to us!
                al.append(self.data[col])
        # once we have all arguments, calculate the output
        # figure out if axis is native to dat, or a scalar, or a function
        if method is None:
            return self.data[self.datCols[a]]
        elif isinstance(method, int) or isinstance(method, float):
            return method * al[0]
        elif isinstance(method, object):
            return method(*al)
        else:
            print 'could not calculate axis for key {0}'.format(key)

    def _gengrid(self, xlis=None, ylis=None, fill_value=None):
        """
            generate regularly spaced y and x bins to use for gridding 2d data
            grid_factor:  multiplier factor for blowing up grid
            grid all input channels (ai0-ai3) to the set xi and yi attributes
        """
        grid_factor = self.grid_factor
        #if xygrid is already properly set, skip filters and generate the grid
        if xlis is not None:
            self.xi = xlis
        else:
            #if no global axes steps and bounds are defined, find them based on data
            #generate lists from data
            #xlis = sorted(self.data[self.xcol])
            xlis = sorted(self.xy[0])
            xtol = self.units[self.xvar][1]
            # values are binned according to their averages now, so min and max 
            #  are better represented
            xstd = []
            xs = []
            # check to see if unique values are sufficiently unique
            # deplete to list of values by finding points that are within 
            #  tolerance
            while len(xlis) > 0:
                # find all the xi's that are like this one and group them
                # after grouping, remove from the list
                set_val = xlis[0]
                xi_lis = [xi for xi in xlis if np.abs(set_val - xi) < xtol]
                # the complement of xi_lis is what remains of xlis, then
                xlis = [xi for xi in xlis if not np.abs(xi_lis[0] - xi) < xtol]
                xi_lis_average = sum(xi_lis) / len(xi_lis)
                xs.append(xi_lis_average)
                xstdi = sum(np.abs(xi_lis - xi_lis_average)) / len(xi_lis)
                xstd.append(xstdi)
            tol = sum(xstd) / len(xstd)
            # create uniformly spaced x and y lists for gridding
            # infinitesimal offset used to properly interpolate on bounds; can
            #   be a problem, especially for stepping axis
            tol = max(tol, 1e-1)
            self.xi = np.linspace(min(xs)+tol,max(xs)-tol,
                                  num=(len(xs) + (len(xs)-1)*(grid_factor-1)))
        if ylis is not None:
            self.yi = ylis
        else:
            #ylis = sorted(self.data[self.ycol])
            ylis = sorted(self.xy[1])
            ytol = self.units[self.yvar][1]
            ystd = []
            ys = []
            while len(ylis) > 0:
                set_val = ylis[0]
                yi_lis = [yi for yi in ylis if np.abs(set_val - yi) < ytol]
                ylis = [yi for yi in ylis if not np.abs(yi_lis[0] - yi) < ytol]
                yi_lis_average = sum(yi_lis) / len(yi_lis)
                ys.append(yi_lis_average)
                ystdi = sum(np.abs(yi_lis - yi_lis_average)) / len(yi_lis)
                ystd.append(ystdi)
            tol = sum(ystd) / len(ystd)
            tol = max(tol, 1e-1)
            self.yi = np.linspace(min(ys)+tol,max(ys)-tol,
                                  num=(len(ys) + (len(ys)-1)*(grid_factor-1)))
            

        x_col, y_col = self.xy 
        # grid each of our signal channels
        for key in self.zvars:
            zcol = self.datCols[key]            
            #make fill value znull right now (instead of average value)
            fill_value = self.znull #self.data[zcol].sum()  / len(self.data[zcol])
            grid_i = griddata((x_col,y_col), self.data[zcol], 
                               (self.xi[None,:],self.yi[:,None]),
                                method='linear',fill_value=fill_value)
            self.zvars[key] = grid_i
        self.zi = self.zvars[self.zvar]

    def dist_filter(cutoff=0.95):
        """
        masks data points that fall out of the central distribution of data
        good for removing scatter points
        """
        pass

    def intaxis(self, intVar, filename=None):
         if intVar == self.xvar: #sum over all x values at fixed y
             dataDump = np.zeros((len(self.yi),2))
             for y in range(len(self.yi)):
                 dataDump[y][0] = self.yi[y]
                 dataDump[y][1] = self.zi[y].sum() -  self.znull * len(self.xi)

             np.savetxt(filename, dataDump)

         elif intVar == self.yvar: #sum over all y values at fixed x
             dataDump = np.zeros((len(self.xi),2))
             for x in range(len(self.xi)):
                 dataDump[x][0] = self.xi[x]
                 for y in range(len(self.yi)):
                     dataDump[x][1] += self.zi[y][x] - self.znull

             np.savetxt(filename, dataDump)
            
         else:
             print 'specified axis is not recognized'

    def level(self, npts, axis=None, aggregate=False):
        """
            for wigner scans:  offset delay slices based on specified sample points
            if you can specify which axis is used for baseline with axis kwarg            
            if npts is negative, will sample from the end of the zi slice
            use aggregate if you want to offset all data by the same average--
                as opposed to slice by slice
        """
        # verify npts not zero
        npts = int(npts)
        if npts == 0:
            print 'cannot level if no sampling range is specified'
            return
        delays = ['d1','d2','t2p1','t21','dref']
        begin = None
        end = None
        var = None
        if axis is None:
            baseline = None
            if self.yvar in delays:
                var = self.yvar
                axis = 0
                if npts < 0:
                    begin, end = self.yi[npts], self.yi[-1]
                    baseline = self.zi[npts:,:]
                elif npts > 0:
                    begin, end = self.yi[0], self.yi[npts]
                    baseline = self.zi[ :npts,:]
                if aggregate:
                    offset = baseline.sum() / (np.abs(npts) * self.xi.size)
                    self.zi = self.zi - offset
                    baseline = baseline - offset
                else:
                    offset = baseline.sum(axis=axis) / np.abs(npts)
                    self.zi = self.zi - offset[None,:]
                    baseline = baseline - offset[None,:]
            elif self.xvar in delays:
                var = self.xvar
                axis = 1
                if npts < 0:
                    begin, end = self.xi[npts], self.xi[-1]
                    baseline = self.zi[:,npts:]
                elif npts > 0:
                    begin, end = self.xi[0], self.xi[npts]
                    baseline = self.zi[:, :npts]
                if aggregate:
                    offset = baseline.sum() / (np.abs(npts) * self.yi.size)
                    self.zi = self.zi - offset
                    baseline = baseline - offset
                else:
                    offset = baseline.sum(axis=axis) / np.abs(npts)
                    self.zi = self.zi - offset[:,None]
                    baseline = baseline - offset[:,None]
            else:
                print 'Level failed:  no delay axis was found'
                return
        elif axis == 0 or axis in ['x', self.xvar]:
                var = self.xvar
                axis = 1
                if npts < 0:
                    begin, end = self.xi[npts], self.xi[-1]
                    baseline = self.zi[:,npts:]
                elif npts > 0:
                    begin, end = self.xi[0], self.xi[npts]
                    baseline = self.zi[:, :npts]
                if aggregate:
                    offset = baseline.sum() / (np.abs(npts) * self.yi.size)
                    self.zi = self.zi - offset
                    baseline = baseline - offset
                else:
                    offset = baseline.sum(axis=axis) / np.abs(npts)
                    self.zi = self.zi - offset[:,None]
                    baseline = baseline - offset[:,None]
        elif axis == 1 or axis in ['y', self.yvar]:
                var = self.yvar
                axis = 0
                if npts < 0:
                    begin, end = self.yi[0], self.yi[npts]
                    baseline = self.zi[npts:,:]
                elif npts > 0:
                    begin, end = self.yi[0], self.yi[npts]
                    baseline = self.zi[ :npts,:]
                if aggregate:
                    offset = baseline.sum() / (np.abs(npts) * self.xi.size)
                    self.zi = self.zi - offset
                    baseline = baseline - offset
                else:
                    offset = baseline.sum(axis=axis) / np.abs(npts)
                    self.zi = self.zi - offset[None,:]
                    baseline = baseline - offset[None,:]
        print r'Baselined according to {0}:  {1} - {2} {3} range'.format(
            var, begin, end, self.units[var][2])
        # since we know what portion of the data is baseline, 
        # we can provide an estimate of data noise
        self.variance = (baseline**2).sum() / baseline.size
        print 'Measured noise variance of {0}'.format(self.variance)
        self.znull = 0.
        self.zmin = self.zi.min()
        self.zmax = self.zi.max()
        try:
            self.zvars[self.zvar] = self.zi
        except AttributeError: # not all Dat instances have zvars
            pass

    def normalize(self,ntype=None,
                  x_file=None,y_file=None,
                  xnSigVar=None,ynSigVar=None,
                  xpower=None, ypower=None,
                  old_fit=False,
                  channel=None): 
        """
            A scaling technique to alter either all the pixels uniformly (i.e. 
            a unit conversion), or certain pixels based on their x and y values.
            channel can be used to normalize zi by a zvars channel
        """
        if ntype is None:
            print 'no ntype selected; normalizing to max amplitude and znull'
            zi_amp = max(np.abs([self.zi.max() - self.znull, self.zi.min() - self.znull]))
            self.zi = (self.zi - self.znull) / zi_amp
            self.zmax = self.zi.max()
            self.zmin = self.zi.min()
            self.znull = 0.
            return
        elif ntype == 'wavelength' or ntype=='b': 
            freq_units = ['nm', 'wn', 'eV']
            x_unit = self.units[self.xvar][2]
            y_unit = self.units[self.yvar][2]
            freqs = ['l1', 'l2', 'l3', 'lm']
            if self.debug:
                plt.figure()
            # output scales as a function of wavelength (optics, opa power, etc.)
            if x_unit in freq_units or y_unit in freq_units:
                # first find x normalization values, then y normalization values
                if x_unit in freq_units:
                    print 'Need normalization file for ',self.xvar,' from ',min(self.xi),' to ',max(self.xi)
                    # import the desired colors file
                    if x_file:
                        x_file_path, x_file_name, x_file_suffix = filename_parse(x_file)
                        if x_file_suffix == 'dat':
                            xNorm = Dat(filepath=x_file, scantype='norm', cols=self.cols)
                            if not xnSigVar:
                                xnSigVar = raw_input('which column has normalization signal (ai1, ai2, ai3)?')
                            xnCol = xNorm.datCols[self.xvar] 
                            xnSigCol = xNorm.datCols[xnSigVar]
                        elif x_file_suffix == 'fit':
                            xNorm = Fit(filepath=x_file, old_cols=old_fit)
                            xnCol = xNorm.cols['set_pt'][0] 
                            xnSigCol = xNorm.cols['amp'][0]
                        try:
                            # convert if in wavenumber units
                            # note:  data[xnCol] values must be in ascending order
                            if self.xvar == 'w1' or self.xvar == 'w2' or self.xvar == 'wm':
                                xNorm.data[xnCol] = 10**7 / xNorm.data[xnCol]
                            # to interpolate, make sure points are ordered by ascending x value
                            xnData = zip(xNorm.data[xnCol],xNorm.data[xnSigCol])
                            xnData.sort()
                            xnData = zip(*xnData)
                            xnData = np.array(xnData)
                            if self.debug:
                                plt.plot(xnData[0],xnData[1],label='xvar')
                            # w2 gets squared for normalization in standard treatment
                            fx = interp1d(xnData[0],xnData[1], kind='cubic', bounds_error=True)
                        except:
                            print 'failed to generate norm function for {0}'.format(self.xvar)
                            fx = False #interp1d([min(self.xi),max(self.xi)],[1,1])
                    # rather than look for a file, don't normalize by x if 
                    # x_file is not given
                    else:
                        print 'no file found for xnorm using filepath {0}'.format(x_file)
                        fx = False
                else:
                    fx = None
                    #xni = np.ones(len(self.xi))

                if self.yvar in freqs:                
                    print 'Need normalization file for ',self.yvar,' from ',min(self.yi),' to ',max(self.yi)
                    #import the desired colors file using a special case of the module!
                    if y_file:
                        y_file_path, y_file_name, y_file_suffix = filename_parse(y_file)
                        if y_file_suffix == 'dat':
                            print 'in here!'
                            yNorm = Dat(filepath=y_file, scantype='norm', cols=self.cols)
                            if not ynSigVar:
                                ynSigVar = raw_input('which column has normalization signal (ai1, ai2, ai3)?')
                            ynCol = yNorm.datCols[self.yvar] 
                            ynSigCol = yNorm.datCols[ynSigVar]
                        elif y_file_suffix == 'fit':
                            yNorm = Fit(filepath=y_file, old_cols=old_fit)
                            ynCol = yNorm.cols['set_pt'][0] 
                            ynSigCol = yNorm.cols['amp'][0]
                        try:
                            if self.yvar == 'w1' or self.yvar == 'w2' or self.yvar == 'wm':
                                yNorm.data[ynCol] = 10**7 / yNorm.data[ynCol]
                            ynData = zip(yNorm.data[ynCol],yNorm.data[ynSigCol])
                            ynData.sort()
                            ynData = zip(*ynData)
                            ynData = np.array(ynData)
                            if self.debug:
                                plt.plot(ynData[0],ynData[1],label='yvar')
                            fy = interp1d(ynData[0],ynData[1], kind='cubic', bounds_error=True)
                        except:
                            print 'failed to generate norm function for {0}'.format(self.yvar)
                            fy = False#interp1d([min(self.yi),max(self.yi)],[1,1])
                            return
                    else:
                        print 'no file found for ynorm using filepath {0}'.format(y_file)
                        fx = False
                    #yni = griddata(ynData[0],ynData[1], self.yi, method='cubic')
                    #fyi = fy(self.yi)
                    #plt.plot(self.yi,fyi)
                else:
                    fy = None

                #normalize by w2 by both beam energies
                # if x and y powers are not given, make a guess
                if xpower is None:
                    if self.xvar == 'w2' or self.xvar == 'l2':
                        xpower = 2
                    else: 
                        xpower = 1
                if ypower is None:
                    if self.yvar == 'w2' or self.yvar == 'l2':
                        ypower = 2
                    else:
                        ypower = 1
                if not self.znull:
                    znull = self.data[self.zcol].min()
                else:
                    znull = self.znull
                # begin normalization of data points
                # after scaling, offset by znull so zero remains the same
                for i in range(len(self.data[self.zcol])):
                    #match data's x value to our power curve's values through interpolation
                    zi = self.data[self.zcol][i]
                    if fx:
                        try:
                            zi = (zi - znull) / (fx(self.data[self.xcol][i])**xpower) + znull
                        except ValueError:
                            #see if value is near bounds (to within tolerance)
                            if np.abs(self.data[self.xcol][i]-xnData[0].max()) < self.units[self.xvar][1]:
                               zi = (zi - znull) / (fx(xnData[0].max())**xpower) + znull
                            elif np.abs(self.data[self.xcol][i]-xnData[0].min()) < self.units[self.xvar][1]:
                                zi = (zi - znull) / (fx(xnData[0].min())**xpower) + znull
                            else:
                                print 'There is a problem with element x={0}, row {1}'.format(self.data[self.xcol][i],i)  
                                print 'norm data has range of: {0}-{1}'.format(xnData[0].min(), xnData[0].max())
                                return
                        except ZeroDivisionError:
                            print 'divided by zero at element x={0}, row {1}'.format(self.data[self.xcol][i],i)  
                            zi = znull
                    if fy:
                        try:
                            zi = (zi - znull) / (fy(self.data[self.ycol][i])**ypower) + znull
                        except ValueError:
                            #see if value is near bounds (to within tolerance)
                            if np.abs(self.data[self.ycol][i]-ynData[0].max()) < self.units[self.yvar][1]:
                                zi = (zi - znull) / (fy(ynData[0].max())**ypower) + znull
                            elif np.abs(self.data[self.ycol][i]-ynData[0].min()) < self.units[self.yvar][1]:
                                zi = (zi - znull) / (fy(ynData[0].min())**ypower) + znull
                            else:
                                print 'There is a problem with element y={0}, row {1}'.format(self.data[self.ycol][i],i)  
                                print 'norm data has range of: {0}-{1}'.format(ynData[0].min(), ynData[0].max())
                                return
                        except ZeroDivisionError:
                                print 'divided by zero at element y={0}, row {1}'.format(self.data[self.ycol][i],i)  
                                zi = znull
                    self.data[self.zcol][i] = zi
                # offset so that znull = 0
                self.data[self.zcol] = self.data[self.zcol] - znull
                self.znull = 0.
                # now interpolate the new data and create a new zi grid
                self._gengrid()
                # do NOT update zmin and zmax unless zmin and zmax were the 
                #  bounds before normalization 
                self.zmax = self.zi.max()
                self.zmin = self.zi.min()

            else:
                print 'wavelength normalization not needed:  x and y vars are wavelength invariant'
        # now for trace-localized normalization
        # ntype specifies the traces to normalize
        # used to be called equalize
        elif ntype in ['horizontal', 'h', 'x', self.xvar]: 
            nmin = self.znull
            #normalize all x traces to a common value 
            maxes = self.zi.max(axis=1)
            numerator = (self.zi - nmin)
            denominator = (maxes - nmin)
            for i in range(self.zi.shape[0]):
                self.zi[i] = numerator[i]/denominator[i]
            self.zmax = self.zi.max()
            self.zmin = self.zi.min()
            self.znull = 0.
            print 'normalization complete!'
        elif ntype in ['vertical', 'v', 'y', self.yvar]: 
            nmin = self.znull
            maxes = self.zi.max(axis=0)
            numerator = (self.zi - nmin)
            denominator = (maxes - nmin)
            for i in range(self.zi.shape[1]):
                self.zi[:,i] = numerator[:,i] / denominator[i]
            self.zmax = self.zi.max()
            self.zmin = self.zi.min()
            self.znull = 0.
            print 'normalization complete!'
        else:
                print 'did not normalize because only programmed to handle linear, log, or power normalization'
    
    def plot2d(self, alt_zi='raw', 
               scantype=None, contour=False, aspect=None, pixelated=False, 
               dynamic_range=False, floor=None, ceiling=None):
        """
            offset is deprecated and should not be used: 
                invoke zmin attribute to shift data values
            dynamic_range will force the colorbar to use all of it's colors
            floor used exclusively for divergent zero signal scales (amp and log)
            floor is the cutoff to be established in the scaled space
            to turn off normalized scale for amp and log alt_zi, set floor and/
            or ceiling.  can set both to "True"
        """
        # delete old plot data stored in the plt class
        plt.close()
        # update parameters
        matplotlib.rcParams.update({
            'font.size':self.font_size
        })
        p1 = plt.figure()
        gs = grd.GridSpec(1,2, width_ratios=[20,1], wspace=0.1)
        # if the plot is a 2d delay or 2d freq, add extra gridlines to guide the eye
        # also, set the aspect ratio so axes have equal measure
        delays = ['d1','d2','t2p1','t21']
        freq = ['w1','w2', 'w3', 'wm']
        energies = ['E1', 'E2', 'E3', 'Em']
        wavelength = ['l1','l2', 'l3', 'lm']
        # better to check if units agree?
        if ((self.xvar in delays and self.yvar in delays) 
            or (self.xvar in freq and self.yvar in freq) 
            or (self.xvar in energies and self.yvar in energies)
            or (self.xvar in wavelength and self.yvar in wavelength)
            or self.units[self.xvar][2] == self.units[self.yvar][2]):
            if aspect:
                s1 = p1.add_subplot(gs[0], aspect=aspect)
            else:
                s1 = p1.add_subplot(gs[0], aspect='equal')
            diag_min = max(min(self.xi),min(self.yi))
            diag_max = min(max(self.xi),max(self.yi))
            plt.plot([diag_min, diag_max],[diag_min, diag_max],'k:')
        else:
            s1 = p1.add_subplot(gs[0])
        # attach to the plot objects so further manipulations can be done
        self.p1=p1
        self.gs=gs
        self.s1=s1

        if alt_zi in ['int', None, 'raw', '+', '-']:
            znull = None
            if alt_zi == '+':
                znull = self.znull
                zi_norm = np.ma.masked_less(self.zi, znull)
                zi_norm = zi_norm.filled(znull)
                lbound = znull
                ubound = self.zmax
            elif alt_zi == '-':
                znull = self.znull
                zi_norm = np.ma.masked_greater(self.zi, znull)
                zi_norm = zi_norm.filled(znull)
                lbound = self.zmin
                ubound = znull
            else:
                if alt_zi == 'int':
                    # for regular normalized (unscaled, normalized to znull-zmax range)
                    # first offset and normalize data
                    z_sign_mag = max(np.abs([self.zmax-self.znull, self.zmin-self.znull]))
                    zi_norm = (self.zi - self.znull) / z_sign_mag
                    znull = 0.
                else: # alt_zi in [None, 'raw']
                    zi_norm = self.zi
                    znull = self.znull
                if self.zmax == self.zi.max():
                    zmax = max(znull, zi_norm.max())
                else:
                    zmax = self.zmax
                if self.zmin == self.zi.min():
                    zmin = min(znull, zi_norm.min())
                else:
                    zmin = self.zmin
                # now I have to whether or not the data is signed, if zmin and zmax
                # are on the same side of znull, then the data only has one sign!
                if znull >= max(zmin, zmax):
                    # data is negative sign
                    print 'data has only negative sign'
                    if dynamic_range:
                        ubound = zmax
                    else:
                        ubound = znull
                    lbound = zmin
                elif znull <= min(zmin, zmax):
                    # data is positive sign
                    print 'data has only positive sign'
                    if dynamic_range:
                        lbound = zmin
                    else:
                        lbound = znull
                    ubound = zmax
                else:
                    # data has positive and negative sign, so center the colorbar
                    print 'data has positive and negative sign'
                    if dynamic_range:
                        # check for whether positive or negative signals extend less
                        # using smaller range on both sides of znull ensures full 
                        # dynamic range of colorbar
                        if -zmin + znull < zmax - znull:
                            ubound = np.abs(zmin)
                        else:
                            ubound = np.abs(zmax)
                    else:
                        # using larger range on both sides of znull ensures full 
                        # range of data will be shown
                        if -zmin + znull < zmax - znull:
                            ubound = np.abs(zmax)
                        else:
                            ubound = np.abs(zmin)
                    lbound = 2*znull - ubound
            print 'lower and upper bounds:', lbound, ubound
            levels = np.linspace(lbound, ubound, num=200)
        elif alt_zi in ['amp', 'log']:
            zi_norm = np.ma.masked_less_equal(
                self.zi - self.znull, 0.)
            if alt_zi == 'amp':
                # for sqrt scale (amplitude)
                zi_norm = np.sqrt(zi_norm)
                if floor is not None:
                    self.floor = floor
                else:
                    self.floor = 0.
                zi_norm = np.ma.masked_less_equal(zi_norm, self.floor)
                zi_norm = zi_norm.filled(self.floor)
                if ceiling is not None:
                    self.ceiling = ceiling
                else:
                    self.ceiling = np.sqrt(self.zmax - self.znull) # zi_norm.max()
                zi_norm = np.ma.masked_greater(zi_norm, self.ceiling)
                zi_norm = zi_norm.filled(self.ceiling)
            elif alt_zi == 'log':
                # for log scale
                zi_norm = np.log10(zi_norm)
                if floor is not None:
                    self.floor = floor
                else:
                    self.floor = zi_norm.min()
                zi_norm = np.ma.masked_less_equal(zi_norm, self.floor)
                zi_norm = zi_norm.filled(self.floor)
                if ceiling is not None:
                    self.ceiling = ceiling
                else:
                    self.ceiling = zi_norm.max()
                zi_norm = np.ma.masked_greater(zi_norm, self.ceiling)
                zi_norm = zi_norm.filled(self.ceiling)
            levels = np.linspace(self.floor, self.ceiling, num=200)
        else:
            print 'alt_zi type {0} not recognized; plotting on raw scale'.format(alt_zi)
            zi_norm = self.zi
            levels = 200 
        self.alt_zi=alt_zi
        self.zi_norm = zi_norm
        # plot the data
        if pixelated:
            # need to input step size to get centering to work
            x_step = np.abs(self.xi[1] - self.xi[0])
            y_step = np.abs(self.yi[1] - self.yi[0])
            if aspect:
                pixel_aspect=aspect
            else:
                # this weighting makes the plot itself square
                pixel_aspect = (self.xi.max() - self.xi.min()) / (self.yi.max() - self.yi.min())
                # this weighting gives square pixels...?
                #pixel_aspect = 1. / pixel_aspect
            cax = plt.imshow(zi_norm, origin='lower', cmap=self.mycm, 
                             interpolation='nearest', 
                             vmin=levels.min(), vmax=levels.max(),
                             extent=[self.xi.min() - x_step/2., 
                                     self.xi.max() + x_step/2., 
                                     self.yi.min() - y_step/2., 
                                     self.yi.max() + y_step/2.])#,
                             #aspect=pixel_aspect)
            plt.gca().set_aspect(pixel_aspect, adjustable='box-forced')
        else:
            cax = plt.contourf(self.xi, self.yi, zi_norm, levels, 
                               cmap=self.mycm)
        self.cax=cax
        if contour:
            # normalize to zmin and zmax if they are smaller than the bounds
            # of the data
            plt.contour(self.xi, self.yi, self.zi_norm, self.contour_n, 
                        **self.contour_kwargs)
        #matplotlib.axes.rcParams.viewitems
        plt.xticks(rotation=45)
        plt.grid(b=True)
        if self.limits:
            v = np.array([self.xlim[0], self.xlim[1],
                          self.ylim[0], self.ylim[1]])
        else:
            v = np.array([self.xi.min(), self.xi.max(),
                          self.yi.min(), self.yi.max()])
            #x_decimal=max(0,int(np.ceil(np.log10(np.abs(v[1]-v[0]))+2)))
            #y_decimal=max(0,int(np.ceil(np.log10(np.abs(v[3]-v[2]))+2)))
            #v[0:2] = np.around(v[0:2], decimals=x_decimal)
            #v[2:] = np.around(v[2:], decimals=y_decimal)
        s1.axis(v)

        if aspect:
            s1.set_aspect(aspect)
        # window the plot; use either 2d plot dimensions or set window
        plt.ylabel(self.units[self.yvar][3], fontsize=self.font_size)
        plt.xlabel(self.units[self.xvar][3], fontsize=self.font_size)
        p1.subplots_adjust(bottom=0.18)
        #s1.set_adjustable('box-forced')
        s1.autoscale(False)
        print 'plotting finished!'

    def savefig(self, fname=None, **kwargs):
        """
            generates the image file by autonaming the file
            default image type is 'png'
        """
        if self.p1:        
            pass
        else:
            print 'no plot is associated with the data. cannot save'
            return
        if fname is None:
            fname = self.filename
            filepath = self.filepath
            file_suffix = 'png'
        else:
            filepath, fname, file_suffix = filename_parse(fname)
            if not file_suffix:
                file_suffix = 'png' 
        if 'transparent' not in kwargs:
            kwargs['transparent'] = True
        if filepath:
            fname = filepath + '\\' + fname
        fname = find_name(fname, file_suffix)
        fname = fname + '.' + file_suffix
        self.p1.savefig(fname, **kwargs)
        print 'image saved as {0}'.format(fname)

    def side_plots(self, subplot, 
                    # do we project (bin) either axis?
                    x_proj=True, y_proj=True, 
                    # provide a list of coordinates for sideplot
                    x_list=None, y_list=None,
                    # provide a NIRscan object to plot
                    x_obj=None, y_obj=None):
        """
            position complementary axis plot on x and/or y axes of subplot
            new:  side_plots now project the plot scale (alt_zi and zi_norm), 
            instead of simply the stored zi array
        """
        #if there is no 1d_object, try to import one
        divider = make_axes_locatable(subplot)
        if x_proj or x_list or x_obj:
            axCorrx = divider.append_axes('top', 0.75, pad=0.3, sharex=subplot)
            axCorrx.autoscale(False)
            axCorrx.set_adjustable('box-forced')
            # make labels invisible
            plt.setp(axCorrx.get_xticklabels(), visible=False)
            axCorrx.get_yaxis().set_visible(False)
            axCorrx.grid(b=True)
        if y_proj or y_list or y_obj:
            axCorry = divider.append_axes('right', 0.75, pad=0.3, sharey=subplot)
            axCorry.autoscale(False)
            axCorry.set_adjustable('box-forced')
            # make labels invisible
            plt.setp(axCorry.get_yticklabels(), visible=False)
            axCorry.get_xaxis().set_visible(False)
            axCorry.grid(b=True)
        if x_proj:
            # integrate the axis
            if self.alt_zi == 'log':
                x_ax_int = self.zi_norm.sum(axis=0) - self.floor * len(self.yi)
            else:
                x_ax_int = self.zi_norm.sum(axis=0) - self.znull * len(self.yi)
            # normalize (min is a pixel)
            xmax = max(np.abs(x_ax_int))
            x_ax_int = x_ax_int / xmax
            axCorrx.plot(self.xi,x_ax_int,self.side_plot_proj_linetype,
                         **self.side_plot_proj_kwargs)
            if min(x_ax_int) < 0:
                axCorrx.set_ylim([-1.1,1.1])
                axCorrx.plot([self.xi.min(), self.xi.max()],[0,0],'k:')
            else:
                axCorrx.set_ylim([0,1.1])
            axCorrx.set_xlim([self.xi.min(), self.xi.max()])
        if y_proj:
            # integrate the axis
            if self.alt_zi == 'log':
                y_ax_int = self.zi_norm.sum(axis=1) - self.floor * len(self.xi)
            else:
                y_ax_int = self.zi_norm.sum(axis=1) - self.znull * len(self.xi)                
            # normalize (min is a pixel)
            ymax = max(np.abs(y_ax_int))
            y_ax_int = y_ax_int / ymax
            axCorry.plot(y_ax_int,self.yi,self.side_plot_proj_linetype,
                         **self.side_plot_proj_kwargs)
            if min(y_ax_int) < 0:
                axCorry.set_xlim([-1.1,1.1])
                axCorry.plot([0,0], [self.yi.min(), self.yi.max()],'k:')
            else:
                axCorry.set_xlim([0,1.1])
            axCorry.set_ylim([self.yi.min(), self.yi.max()])
        if isinstance(x_list, np.ndarray): 
            print x_list.shape
            axCorrx.plot(x_list[0],x_list[1], self.side_plot_else_linetype,
                         **self.side_plot_else_kwargs)
            axCorrx.set_ylim([0.,1.1])
        elif x_obj:
            try:
                x_list = x_obj.data[0][2].copy()
            except IndexError:
                print 'Import failed--data type was not recognized'
            # spectrometer has units of nm, so make sure these agree
            if self.xvar in ['w1','w2','wm']:
                x_list[0] = 10**7 / x_list[0]
            #normalize the data set
            x_list_max = x_list[1].max()
            x_list[1] = x_list[1] / x_list_max
            axCorrx.plot(x_list[0],x_list[1], self.side_plot_else_linetype,
                         **self.side_plot_else_kwargs)
            #axCorrx.set_ylim([0.,1.1])
            axCorrx.set_xlim([self.xi.min(), self.xi.max()])
        if isinstance(y_list, np.ndarray):
            axCorry.plot(y_list[1],y_list[0], self.side_plot_else_linetype,
                         **self.side_plot_else_kwargs)
        elif y_obj:
            try:
                y_list = y_obj.data[0][2].copy()
            except IndexError:
                print 'Import failed--data type was not recognized'
            if self.yvar in ['w1','w2','wm']:
                y_list[0] = 10**7 / y_list[0]
            #normalize the data set
            y_list_max = y_list[1].max()
            y_list[1] = y_list[1] / y_list_max
            axCorry.plot(y_list[1],y_list[0], self.side_plot_else_linetype,
                         **self.side_plot_else_kwargs)
            #axCorry.set_xlim([0.,1.1])
            axCorry.set_ylim([self.yi.min(), self.yi.max()])
    
    def smooth(self, 
               x=0,y=0, 
               window='kaiser'): #smoothes via adjacent averaging            
        """
            convolves the signal with a 2D window function
            currently only equipped for kaiser window
            'x' and 'y', both integers, are the nth nearest neighbor that get 
                included in the window
            Decide whether to perform xaxis smoothing or yaxis by setting the 
                boolean true
        """
        # n is the seed of the odd numbers:  n is how many nearest neighbors 
        # in each direction
        # make sure n is integer and n < grid dimension
        # account for interpolation using grid factor
        nx = x*self.grid_factor
        ny = y*self.grid_factor
        # create the window function
        if window == 'kaiser':
            # beta, a real number, is a form parameter of the kaiser window
            # beta = 5 makes this look approximately gaussian in weighting 
            # beta = 5 similar to Hamming window, according to numpy
            # over window (about 0 at end of window)
            beta=5.0
            wx = np.kaiser(2*nx+1, beta)
            wy = np.kaiser(2*ny+1, beta)
        # for a 2D array, y is the first index listed
        w = np.zeros((len(wy),len(wx)))
        for i in range(len(wy)):
            for j in range(len(wx)):
                w[i,j] = wy[i]*wx[j]
        # create a padded array of zi
        # numpy 1.7.x required for this to work
        temp_zi = np.pad(self.zi, ((ny,ny), 
                                   (nx,nx)), 
                                    mode='edge')
        from scipy.signal import convolve
        out = convolve(temp_zi, w/w.sum(), mode='valid')
        if self.debug:
            plt.figure()
            sp1 = plt.subplot(131)
            plt.contourf(self.zi, 100)
            plt.subplot(132, sharex=sp1, sharey=sp1)
            plt.contourf(w,100)
            plt.subplot(133)
            plt.contourf(out,100)
        self.zi=out
        # reset zmax
        self.zmax = self.zi.max()
        self.zmin = self.zi.min()

    def svd(self):
        """
            singular value decomposition of gridded data z
        """
        U,s,V = np.linalg.svd(self.zi)
        self.U, self.s, self.V = U,s,V
        #give feedback on top (normalized) singular values
        return U, s, V

    def svd_filter(self, variance=None, residual=False):
        """
            filter data using SVD partial sum
            outputs normalized root mean stdev (NRMSD)
            variance should be used to suggest what svd contributions are 
            statistically significant
            residual:  if True, resdiual array is returned as secondary output
        """
        # (re-)initialize the svd array
        self.svd()
        zfit=False
        # determine if limits need to be reset--only done if zmin and zmax are 
        # set to array bounds
        if self.zmin == self.zi.min() and self.zmax == self.zi.max():
            zfit = True
        if variance is None:
            try: variance = self.variance
            except NameError:
                print 'variance is not known: cannot filter'
                return
        index = None
        i = np.arange(len(self.s))
        cutoff = variance * (self.U.shape[0]-i) * (self.V.shape[0]-i)
        frob_norm = 0
        for ii in i:
            print ii, self.s[ii]**2, cutoff[ii]
            frob_norm = np.sqrt((self.s[:ii+1]**2).sum() / (self.s**2).sum())
            print 'Frob norm {0} %'.format(round(frob_norm,4))
            if cutoff[ii] > self.s[ii]**2:
                # don't include this point
                index = ii
                break
        if self.debug:
            plt.figure()
            plt.scatter(range(len(self.s)),self.s)
            plt.plot([0,len(self.s)],[cutoff, cutoff],'k:', linewidth=2)
        # if cutoff never caused a break, data is full rank
        if index is None:
            index = i[-1]
            print 'svd filter failed:  noise is too large'
        self.rank = index
        print 'Data is of rank {0}'.format(self.rank)
        # now execute the partial sum
        temp_sum = np.zeros(self.zi.shape)
        for i in range(self.rank):
            temp_sum += self.U[:,i][:,None] * self.V[i][None,:] * self.s[i]
        # rms error; 
        res = temp_sum - self.zi
        # rms deviation
        RMSD = np.sqrt((res**2).sum()/self.zi.size)
        # normalized rms deviation
        NRMSD = RMSD / (self.zi.max() - self.zi.min())
        # we could calculate average percent error accurately by excluding 
        # points that are 0 to within noise levels
        # normalized by size of zi
        print 'RMSD {0:.2} '.format(RMSD)
        print 'NRMSD {0:.2%}'.format(NRMSD)
        self.zi = temp_sum
        self.zmin, self.zmax = self.zi.min(), self.zi.max()
        if zfit:  self.zfit()
        if residual:
            return NRMSD, residual
        else:
            return NRMSD

    def T(self):
        """
        transpose the matrix and get the x and y axes to follow suit
        """
        self.zi = self.zi.T

        tempxi = self.xi.copy()
        tempyi = self.yi.copy()
        tempxvar = self.xvar.copy()
        tempyvar = self.yvar.copy()

        self.xi = tempyi
        self.yi = tempxi
        
        self.xvar = tempyvar
        self.yvar = tempxvar
        
        print 'x axis is now {0}, and y is {1}'.format(self.xvar, self.yvar)

    def trace(self, val=None, kind=None, save=False):
        """
            returns a 1D trace of the data where val is constant
            both arguments and values of the trace are returned in the format 
                np.array([arg, value])
            val is a constraint that defines the 1D trace
            kind has several options:
                'x':  horizontal trace at fixed y val
                'y':  vertical trace at fixed x val
                'ps':  peak shift parameterized against coherence time (tau)
                    at fixed population time val=T
                    needs to be 2d delay scan
                '3pepsZQC':  peak shift parameterized against coherence time 
                    (tau) at fixed zqc evolution time (pathways 1-3) val
                'diag':  diagonal slice of an equal axis space (eg. 2d freq)
                    with offset val (x - y = offset)
        """
        if kind == 'x':
            # for each slice along y, find y=val
            y = np.zeros(self.zi.shape[1])
            x = self.xi.copy()
            for i in range(self.zi.shape[1]):
                fx = interp1d(self.yi, self.zi[:,i], fill_value=np.nan, 
                              bounds_error=False)
                y[i] = fx(val)
            return x, y
        elif kind == 'y':
            # for each slice along x, pick the value at x=val
            y = np.zeros(self.zi.shape[0])
            x = self.yi.copy()
            for i in range(self.zi.shape[0]):
                fx = interp1d(self.xi, self.zi[i], fill_value=np.nan, 
                              bounds_error=False)
                y[i] = fx(val)
            return x, y
            """
            elif kind in ['ps', '3peps']:
                trace = self._3PEPS_trace(val, w2w2p_pump=True)
                savestr = '{0}3peps \\{1}.T{2}.txt'.format(self.savepath,self.filename,val)
            elif kind in ['ps-zqc', 'zqc', '3peps-zqc']:
                trace = self._3PEPS_trace(val, w2w2p_pump=False)
                savestr = '{0}3peps-zqc \\{1}.T{2}.txt'.format(self.savepath,self.filename,val)
            """
        elif kind == 'diag':
            if val:
                offset = val
            else: offset = 0.0
            # just pick an axis to loop over and go with it
            for i in range(self.zi.shape[1]):
                val = self.xi[i] + offset
                if val < self.yi.max() and val > self.yi.min():
                    fx = interp1d(self.yi, self.zi[:,i], fill_value=np.nan, 
                                  bounds_error=False)
                    y[i] = fx(val)
                else:
                    y[i] = np.nan
            return x,y
        """
        if save:
            np.savetxt(savestr,trace.T)
        """

    def zfit(self):
        """
        redefine the limits of the z array according to min and max
        """
        self.zmax = self.zi.max()
        self.zmin = self.zi.min()

    def z_offset(self, amount, var=None):
        # a method of the Wigner subclass
        # offset data based on their average values at certain delays
        # figure out which axis is temporal
        pass
    
        
class NIRscan:
    #this module has yet to be defined, but will handle typical abs scans
    #functions should be able to plot absorbance spectra as well as normalized 2nd derivative (after smoothing)
    font_size = 16

    def __init__(self):
        self.data = list()
        self.unit = 'nm'
        self.xmin = None
        self.xmax = None

    def add(self, filepath=None,dataName=None):
        #import data file--right now designed to be a file from Rob's spectrometer
        #filepath must yield a file
        #create a list of dictionaries?
        #each file data is inserted as a numpy array into the list data ~ [[name, numpyarray],[name, numpy array]]
        if filepath:
            pass
        else:
            filepath = raw_input('Please enter the filepath:')
        if type(filepath) == str:
            pass
        else:
            print 'Error:  filepath needs to be a string'
            return
        
        if os.path.isfile(filepath):
            print 'found the file!'
        else:
            print 'Error: filepath does not yield a file'
            return

        #is the file suffix one that we expect?  warn if it is not!
        filesuffix = os.path.basename(filepath).split('.')[-1]
        if filesuffix != 'txt':
            should_continue = raw_input('Filetype is not recognized and may not be supported.  Continue (y/n)?')
            if should_continue == 'y':
                pass
            else:
                print 'Aborting'
                return

        
        #create a string that will refer to this list
        if dataName:
            pass
        else:
            dataName = raw_input('Please name this data set:  ')
        #now import file as a local var--18 lines are just txt and thus discarded
        rawDat = np.genfromtxt(filepath, skip_header=18)
        dataSet = [dataName, 'nm', np.zeros((2,len(rawDat)))]
        #store the data in the data array--to be indexed as [variable][data]
        for i in range(len(rawDat)):
            for j in range(2):
                dataSet[2][j][i] = float(rawDat[i][j])
        self.data.append(dataSet)
        if max(self.data[-1][2][0]) > self.xmax:
            self.xmax = max(self.data[-1][2][0])
        if min(self.data[-1][2][0]) < self.xmin:
            self.xmin = min(self.data[-1][2][0])
        print 'file has imported!'
    
    def A(self, scan_no=0, units='wn'):
        from scipy.interpolate import interp1d
        if units=='wn':
            x = 1e7 / np.array(self.data[scan_no][2][0])
            y = self.data[scan_no][2][1]
        elif units== 'nm':
            x = self.data[scan_no][2][0][::-1]
            y = self.data[scan_no][2][1][::-1]
        fx = interp1d(x,y)
        return fx
        
    def plot(self, scantype='default', xtype='wn'):
        self.ax1 = plt.subplot(211)
        matplotlib.rcParams.update({
            'font.size':self.font_size
        })
        for i in range(len(self.data)):
            plotData = self.data[i][2]
            name = self.data[i][0]
            if xtype == 'wn':
                if self.data[i][1] != 'wn':
                    plotData = self._switchUnits(plotData[0],plotData[1])
            elif xtype == 'nm':
                if self.data[i][1] != 'nm':
                    plotData = self._switchUnits(plotData[0],plotData[1])
            self.ax1.plot(plotData[0], plotData[1], label=name)
        plt.ylabel('abs (a.u.)')
        self.ax1.legend(loc=4)
        self.ax1.grid(b=True)
        #now plot 2nd derivative
        for i in range(len(self.data)):
            self.ax2 = plt.subplot(212, sharex=self.ax1)
            preData = self.data[i][2]
            preData = self._smooth(preData)
            name = self.data[i][0]
            #compute second derivative
            plotData = np.array([10**7 / preData[0][:-2], np.diff(preData[1], n=2)])
            #plotData[1] = plotData[1] / (np.diff(preData[0])[:-1]**2)
            #normalize for comparisons of different scans
            #Max = max(max(plotData[1]),-min(plotData[1]))
            #plotData[1] = plotData[1] / Max
            #plot the data!
            self.ax2.plot(plotData[0], plotData[1], label=name)
        self.ax2.grid(b=True)
        plt.xlabel(r'$\bar\nu / cm^{-1}$')

    def _switchUnits(self, xset, yset):
        #converts from wavenumbers to nm and vice-versa
        #sorts data by ascending x values
        xset = 10**7 / xset
        xypairs = zip(xset, yset)
        xypairs.sort()
        return zip(*xypairs)
        
    def _smooth(self, dat1, n=20, window_type='default'):
        #data is an array of type [xlis,ylis]        
        #smooth to prevent 2nd derivative from being noisy
        for i in range(n, len(dat1[1])-n):
            #change the x value to the average
            window = dat1[1][i-n:i+n].copy()
            dat1[1][i] = window.mean()
        return dat1[:][:,n:-n]
    def export(self):
        #write a file with smoothed 2nd derivative data included
        print 'in progress!'

class Fit:
    # old_cols used before COLORS support for extra mixers (~November 2013 and
    # earlier)
    old_cols = {
        'num':  [0],
        'set_pt':   [1],
        'd1':   [2],
        'c1':   [3],
        'd2':   [4],
        'c2':   [5],
        'm1':   [6],
        'mu':   [7],
        'amp':   [8],
        'sigma': [9],
        'gof':   [10]
    }
    cols = {
        'num':  [0],
        'set_pt':   [1],
        'd1':   [2],
        'c1':   [3],
        'd2':   [4],
        'c2':   [5],
        'm1':   [6],
        'm2':   [7],
        'm3':   [8],
        'mu':   [9],
        'amp':   [10],
        'sigma': [11],
        'gof':   [12],
        'mismatch': [13]
    }

    def __init__(self, filepath=None, old_cols=False):
        """
            import a fit file
        """
        if filepath:
            pass
        else:
            filepath = raw_input('Please give the absolute file location:')
        #filepath must yield a file
        if os.path.isfile(filepath):
            self.has_data=True
            print 'found the file!'
        else:
            self.has_data = False
            print 'filepath',filepath,'does not yield a file'
            return
        self.filepath, self.filename, self.file_suffix = filename_parse(filepath)
        rawDat = np.genfromtxt(filepath,dtype=np.float) 
        # define arrray used for transposing dat file so cols are first index
        self.data = rawDat.T
        if old_cols:
            self.cols = self.old_cols
        print 'file has imported'

    def gengrid(self, xlis=None, ylis=None, fill_value=None):
        """
            generate regularly spaced y and x bins to use for gridding 2d data
            grid_factor:  multiplier factor for blowing up grid
            grid all input channels (ai0-ai3) to the set xi and yi attributes
        """
        grid_factor = self.grid_factor
        if xlis is not None:
            # generate the xi vars based on step sizes
            pass
        else:
            #if no global axes steps and bounds are defined, find them based on data
            #generate lists from data
            #xlis = sorted(self.data[self.xcol])
            xlis = sorted(self.xy[0])
            xtol = 1 # motortune should be within 1ustep always
            # values are binned according to their averages now, so min and max 
            #  are better represented
            xstd = []
            xs = []
            # check to see if unique values are sufficiently unique
            # deplete to list of values by finding points that are within 
            #  tolerance
            while len(xlis) > 0:
                # find all the xi's that are like this one and group them
                # after grouping, remove from the list
                set_val = xlis[0]
                xi_lis = [xi for xi in xlis if np.abs(set_val - xi) < xtol]
                # the complement of xi_lis is what remains of xlis, then
                xlis = [xi for xi in xlis if not np.abs(xi_lis[0] - xi) < xtol]
                xi_lis_average = sum(xi_lis) / len(xi_lis)
                xs.append(xi_lis_average)
                xstdi = sum(np.abs(xi_lis - xi_lis_average)) / len(xi_lis)
                xstd.append(xstdi)
            tol = sum(xstd) / len(xstd)
            # create uniformly spaced x and y lists for gridding
            # infinitesimal offset used to properly interpolate on bounds; can
            #   be a problem, especially for stepping axis
            tol = max(tol, 1e-6)
            self.xi = np.linspace(min(xs)+tol,max(xs)-tol,
                                  num=(len(xs) + (len(xs)-1)*(grid_factor-1)))
        if ylis is not None:
            self.yi = ylis
        else:
            #ylis = sorted(self.data[self.ycol])
            ylis = sorted(self.xy[1])
            ytol = 1
            ystd = []
            ys = []
            while len(ylis) > 0:
                set_val = ylis[0]
                yi_lis = [yi for yi in ylis if np.abs(set_val - yi) < ytol]
                ylis = [yi for yi in ylis if not np.abs(yi_lis[0] - yi) < ytol]
                yi_lis_average = sum(yi_lis) / len(yi_lis)
                ys.append(yi_lis_average)
                ystdi = sum(np.abs(yi_lis - yi_lis_average)) / len(yi_lis)
                ystd.append(ystdi)
            tol = sum(ystd) / len(ystd)
            tol = max(tol, 1e-6)
            self.yi = np.linspace(min(ys)+tol,max(ys)-tol,
                                  num=(len(ys) + (len(ys)-1)*(grid_factor-1)))

        x_col, y_col = self.xy 
        # grid each of our signal channels
        for key in self.zvars:
            zcol = self.datCols[key]            
            #make fill value znull right now (instead of average value)
            fill_value = self.znull #self.data[zcol].sum()  / len(self.data[zcol])
            grid_i = griddata((x_col,y_col), self.data[zcol], 
                               (self.xi[None,:],self.yi[:,None]),
                                method='cubic',fill_value=fill_value)
            self.zvars[key] = grid_i
        self.zi = self.zvars[self.zvar]
        
def makefit(**kwargs):
    """
    make a fit file filling in only the arguments specified
    kwargs must be lists or arrays of uniform size and 1D shape
    """
    n = len(kwargs.values()[0])
    out = np.zeros((n, 12))
    #first column is just row number (unless overwritten)
    out[:, 0] = range(n)
    for name, value in kwargs.items():
        #all kwargs have to be the same length to make an intelligable array
        if len(value) == n:
            if name in Fit.cols.keys():
                out[:, Fit.cols[name][0]] = value
            else:
                print 'name {0} is not an appropriate column name'.format(name)
                return
        else:
            print 'Error: not all columns are the same length:  len({0})={1}, len({2}) = {3}'.format(
                kwargs.keys()[0], n, name, len(value))
            return
    return out

def find_name(fname, suffix):
    """
    save the file using fname, and tacking on a number if fname already exists
    iterates until a unique name is found
    returns False if the loop malfunctions
    """
    good_name=False
    # find a name that isn't used by enumerating
    i = 1
    while not good_name:
        try:
            with open(fname+'.'+suffix):
               # file does exist
               # see if a number has already been guessed
               if fname.endswith(' ({0})'.format(i-1)):
                   # cut the old off before putting the new in
                   fname = fname[:-len(' ({0})'.format(i-1))]
               fname += ' ({0})'.format(i)
               i = i + 1
               # prevent infinite loop if the code isn't perfect
               if i > 100:
                   print 'didn\'t find a good name; index used up to 100!'
                   fname = False
                   good_name=True
        except IOError:
            # file doesn't exist and is safe to write to this path
            good_name = True
    return fname

def make_tune(obj, set_var, fname=None, amp='int', center='exp_val', fit=True,
              offset=None, write=True):
    """
        function for turning dat scans into tune files using exp value

        takes a dat class object and transforms it into a fit file

        need to specify which axis we need the expectation value from 
        (set_var; either 'x' or 'y'; the other axis will be called int_var)

        amp can measure either amplitude or integrated itensity

        offset:  the a point contained within the set_var range that you wish 
        to be the zero point--if such a point is included, the exp_values will
        be shifted relative to it.  This is convenient in tunetests if you want 
        to have a specific color you want to set zero delay to.
    """
    if set_var not in ['x', 'y', obj.xvar, obj.yvar]:
        print 'Error:  set_var type not supported: {0}'.format(set_var)
    # make sure obj type is appropriate and extract properties
    #zimin = obj.zi.min()
    tempzi = obj.zi - obj.znull
    if set_var in ['y', obj.yvar]:
        int_var = obj.xvar
        set_var = obj.yvar
        set_lis = obj.yi
        #int_lis = obj.xi
        axis = 1
    elif set_var in ['x', obj.xvar]:
        int_var = obj.yvar
        set_var = obj.xvar
        set_lis = obj.xi
        #int_lis = obj.yi
        axis = 0

    # decide what tune type this is
    # if exp value var is delay, call this zerotune, if mono, call it colortune
    if int_var in ['lm', 'wm']:
        fit_type = 'colortune'
    elif int_var in ['d1', 'd2']:
        fit_type = 'zerotune'
    else:
        # not sure what type of fit it is
        fit_type = 'othertune'
    if fit:
        # use least squares fitting to fill in tune values
        plsq = obj.fit_gauss(axis=set_var)
        obj_amp, obj_exp, obj_width, obj_y0 = plsq
    else:
        # use expectation values and explicit measurements to extract values
        # calculate the expectation value to get the peak center
        obj_exp = obj.center(axis=set_var, center=center)
        # calculate the width of the feature using the second moment
        obj_width = obj.exp_value(axis=set_var, moment=2)
        obj_width = np.sqrt(np.abs(obj_width - obj_exp**2))
        # also include amplitude
        if amp == 'int':
            # convert area to max amplitude assuming gaussian form factor
            obj_amp = obj.exp_value(axis=set_var, moment=0, norm=False)
            obj_amp = obj_amp / (np.sqrt(2*np.pi)* obj_width)
        elif amp == 'max':
            obj_amp = tempzi.max(axis=axis) - obj.znull
    # convert obj_width from stdev to fwhm
    obj_width *= 2*np.sqrt(2*np.log(2))
    # offset the values if specified
    if offset is not None:
        f_exp = interp1d(set_lis,obj_exp, kind='linear')
        off = f_exp(offset)
        obj_exp = obj_exp - off
    # convert color to nm for fit file
    if set_var in ['w1', 'w2', 'wm']:
        set_lis = 10**7 / set_lis
    # put wavelength in ascending order
    pts = zip(set_lis, obj_exp, obj_amp)
    pts.sort()
    pts = zip(*pts)
    set_lis, obj_exp, obj_amp = pts
    out = makefit(set_pt=set_lis, mu=obj_exp, amp=obj_amp, sigma=obj_width)
    # make a fit file using the expectation value data
    # first, make sure fname has proper format 
    # append descriptors to filename regardless of whether name is provided
    # emulates how COLORS treats naming
    if fit:
        auto = '{0} {1} fitted'.format(set_var, fit_type)
    elif center == 'exp_val':
        auto = '{0} {1} exp_value center'.format(set_var, fit_type)
    elif center == 'max':
        auto = '{0} {1} max value center'.format(set_var, fit_type)
    else:
        auto = '{0} {1}'.format(set_var, fit_type)
    # suffix:  let me add the .fit filename suffix
    if fname is not None:
        filepath, fname, filesuffix = filename_parse(fname)
        # path:  don't imply path if an absolute path is given
        fname = ' '.join([fname, auto])
        if filepath is None:
            filepath=obj.filepath
    else:
        # use object's filepath as default
        filepath = obj.filepath
        fname = auto
    
    if filepath is not None:
        fname = filepath + '\\' + fname
    fstr = find_name(fname, 'fit')
    if not fstr:
        print 'Could not write file without overwriting an existing file'
        print 'Aborting file write'
        return
    with file(fstr+'.fit', 'a') as exp_file:
        np.savetxt(exp_file, out, delimiter='\t', fmt='%.3f')
    print 'saved as {0}'.format(fstr+'.fit')

def filename_parse(fstr):
    """
    parses a filepath string into it's path, name, and suffix
    """
    split = fstr.split('\\')
    if len(split) == 1:
        file_path = None
    else:
        file_path = '\\'.join(split[0:-1])
    split2 = split[-1].split('.')
    # try and guess whether a suffix is there or not
    # my current guess is based on the length of the final split string
    # suffix is either 3 or 4 characters
    if len(split2[-1]) == 3 or len(split2[-1]) == 4:
        file_name = '.'.join(split2[0:-1])
        file_suffix = split2[-1]
    else:
        file_name = split[-1]
        file_suffix = None
    return file_path, file_name, file_suffix    
    
def gauss_residuals(p, y, x):
    """
    calculates the residual between y and a gaussian with:
        amplitude p[0]
        mean p[1]
        stdev p[2]
    """
    A, mu, sigma, offset = p
    # force sigma to be positive
    err = y-A*np.exp(-(x-mu)**2 / (2*np.abs(sigma)**2)) - offset
    return err

class SVD:
    """
    structure for easy manipulations of SVD objects
    """
    def __init__():
        pass
