# -*- coding: utf-8 -*-
"""
Created on Tue Aug 04 01:47:06 2015

store cmap library for fscolors

@author: Dan
"""
from matplotlib.colors import LinearSegmentedColormap
import numpy as np

def cubehelix_r(gamma=1.0, s=0.5, r=-1.5, h=1.0):
    def get_color_function(p0, p1):
        def color(x):
            # Apply gamma factor to emphasise low or high intensity values
            xg = x ** gamma

            # Calculate amplitude and angle of deviation from the black
            # to white diagonal in the plane of constant
            # perceived intensity.
            a = h * xg * (1 - xg) / 2

            phi = 2 * np.pi * (s / 3 + r * x)
            out = xg + a * (p0 * np.cos(phi) + p1 * np.sin(phi))

            return out[::-1]
        return color
    return {
            'red': get_color_function(-0.14861, 1.78277),
            'green': get_color_function(-0.29227, -0.90649),
            'blue': get_color_function(1.97294, 0.0),
    }



def ch_augment(gamma=1.0, s=0.5, r=-1.5, h=0.5,
               lum_rev=False, darkest=0.8):
    # isoluminescent curve--helical color cycle
    rr = (.213/.30)**1#.5
    rg = (.715/.99)**1#.5
    rb = (.072/.11)**1#.5
    def get_color_function(p0, p1):
        def color(x):
            # Apply gamma factor to emphasise low or high intensity values
            #xg = x ** gamma

            # Calculate amplitude and angle of deviation from the black
            # to white diagonal in the plane of constant
            # perceived intensity.
            xg = darkest * x**gamma
            lum = 1-xg # starts at 1
            if lum_rev:
                lum = lum[::-1]
            #a = h*lum * (1-lum)/2.
            #"""
            a = lum.copy()#h * lum*(1-lum)/2.
            a[lum<0.5] = h * lum[lum<0.5]/2.
            a[lum>=0.5] = h * (1-lum[lum>=0.5])/2.
            #"""
            phi = 2 * np.pi * (s / 3 + r * x)
            out = lum + a * (p0 * np.cos(phi) + p1 * np.sin(phi))

            return out
        return color
    return {
            'red': get_color_function(-0.14861*rr, 1.78277*rr),
            'green': get_color_function(-0.29227*rg, -0.90649*rg),
            'blue': get_color_function(1.97294*rb, 0.0),
    }

#cm1_r = cubehelix_r(s=0.5, r=0.9, h=1.4, gamma=0.8)
cm1_r = cubehelix_r(s=0.5, r=0.75, h=1.6, gamma=1.0)
chw1 = LinearSegmentedColormap('dk1', cm1_r)
# iso1 starts on light blue, ends on dark purple
iso1 = ch_augment(gamma=0.5, s=0.25, r=-6/6.,#5.5/6.,#1.,#-2/3., 
            h=1.3, lum_rev=False, darkest=0.7)
# iso2 starts on light blue, ends on dark red
iso2 = ch_augment(gamma=0.5, s=0.25, r=-5/6.,#5.5/6.,#1.,#-2/3., 
            h=1.45, lum_rev=False, darkest=0.75)
# track brightness with the best perceptual color 
# (blue is darkest, then red, and green is lightest)
iso3 = ch_augment(gamma=0.5, s=2.25, r=-5/6.,#5.5/6.,#1.,#-2/3., 
            h=1., lum_rev=False, darkest=0.7)
chw2 = LinearSegmentedColormap('dk2', iso1)
chw3 = LinearSegmentedColormap('dk3', iso2)
chw4 = LinearSegmentedColormap('dk4', iso3)

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
wrightcm = LinearSegmentedColormap.from_list('wright',wrightcolors)
altcm = LinearSegmentedColormap.from_list('signed',signed_cm)
altcm2 = LinearSegmentedColormap.from_list('signed2',blaise_cm)
ampcm = LinearSegmentedColormap.from_list('skye',skye_amp)



def plot_rgb(colorbar):
    import matplotlib.pyplot as plt
    plt.figure()
    x=np.linspace(0,1.,num=256)
    r = colorbar._segmentdata['red'](x)
    r[r>1] = 1
    r[r<0] = 0
    g = colorbar._segmentdata['green'](x)
    g[g>1] = 1
    g[g<0] = 0
    b = colorbar._segmentdata['blue'](x)
    b[b>1] = 1
    b[b<0] = 0
    k = .213*r + .715*g + .072*b
    plt.plot(x,r,'r', linewidth=5, alpha=0.6)
    plt.plot(x,g,'g', linewidth=5, alpha=0.6)
    plt.plot(x,b,'b', linewidth=5, alpha=0.6)
    plt.plot(x,k,'k:', linewidth=5, alpha=0.6)
    
    plt.grid()
