#! /usr/bin/env python
# -*- coding: utf-8 -*-
# vim:fenc=utf-8
#
# Copyright © 2020 Davide Cossu <cossu.dvd@gmail.com>
#
# Distributed under terms of the CC BY-NC license.

"""
  Plotting the Fourier transform and steps
"""

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
import matplotlib.animation as animation

t = np.linspace(0, 4*np.pi, 512)

fig, axs = plt.subplots(2,3, squeeze=True, figsize=(20,29))
xdata, ydata = [],[]
ln, = axs[0,0].plot([], [], 'ro')
ln1, = axs[0,1].plot([], [],'-o', ms=0.2)
ln2, = axs[0,2].plot([], [],'-r', ms=0.2)
ln3, = axs[1,0].plot([], [],'r-', ms=2)
ln4, = axs[1,1].plot([], [],'-r', ms=2)
ln5, = axs[1,2].plot([], [],'-', ms=2)

reData, imData = [], []
avgDataX, avgDataY = [],[]

freqDataX, freqDataY = [],[]


freqs = []

def f(t):
    return 0.5*np.sin(2*t) + 0.25*np.sin(10*t)

def init():
    axs[0,0].set_xlim(0, 4*np.pi)
    axs[0,0].set_ylim(-1, 1)
    axs[0,0].plot(t, f(t))
    #axs[0,0].set_aspect(1)
    axs[0,0].set_title("Function plot")

    axs[0,1].set_xlim(-2.5,2.5)
    axs[0,1].set_ylim(-2.5,2.5)
    axs[0,1].set_aspect(1)
    axs[0,1].plot([1] [0])
    axs[0,1].set_title("Fourier integrand function")

    axs[0,2].set_xlim(-2.5,2.5)
    axs[0,2].set_ylim(-2.5,2.5)
    axs[0,2].set_aspect(1)
    axs[0,2].vlines(x=0,ymin=-2.5,ymax=2.5,colors='k')
    axs[0,2].hlines(y=0,xmin=-2.5,xmax=2.5,colors='k')
    axs[0,2].set_title("Average point position")

    axs[1,0].set_xlim(0,4*np.pi)
    axs[1,0].set_ylim(-0.5,0.5)
    axs[1,0].set_xlabel("Frequency")
    axs[1,0].set_ylabel("x-coordinate of the average point")

    T = 1.0/100.0
    N = 512
    x = np.linspace(0, N*T*np.pi*2, N)
    xs = np.linspace(0, 1.0/(2*T), N//2)
    fourier = np.fft.fft(f(x))
    axs[1,1].plot(xs,
            2.0/N * np.abs(fourier[:256]))
    axs[1,1].set_title("Fast Fourier Transform")

    fs = np.fft.fftfreq(fourier.size, d=x[1]-x[0])
    b = np.abs(fourier[:256])
    a = np.sort(np.abs(fourier[:256]))
    global freqs
    freqs = [
            fs[np.argwhere(b == a[-1])]*np.pi*2,
            fs[np.argwhere(b == a[-10])]*np.pi*2
            ]

    axs[1,2].set_xlim(0,4*np.pi)
    axs[1,2].set_ylim(-1,1)
    axs[1,2].plot(t, np.cos(freqs[0]*t)[0], color='k',ms=0.2)
    axs[1,2].plot(t, np.cos(freqs[1]*t)[0], ls='--',ms=0.2)
    axs[1,2].set_title("Plot of the main frequencies")

    return ln,ln1,ln2,ln3,ln4,ln5

def update(frame):
    xdata.append(frame)
    ydata.append(f(frame))
    ln.set_data(xdata[-1], ydata[-1])
    
    x = 3/(2*np.pi)
    data = f(frame) * np.exp(-2*np.pi*1j*frame*x)
    reData.append(data.real)
    imData.append(data.imag)
    avgDataX.append(np.average(reData))
    avgDataY.append(np.average(imData))

    xi = frame/(2*np.pi)
    tp = np.linspace(0,4*np.pi,512)
    data1 = f(tp) * np.exp(-2*np.pi*1j*tp*xi)
    freqDataY.append(np.average(data1.real))

    freqDataX.append(frame)

    ln3.set_data(freqDataX, freqDataY)
    ln1.set_data(reData, imData)
    ln2.set_data(avgDataX, avgDataY)

    return ln,ln1,ln2,ln3,ln4,ln5

ani = FuncAnimation(fig, update, frames=np.linspace(0, 4*np.pi, 512),
                    init_func=init, blit=True, interval = 10, repeat=False)

def maximize():
    plot_backend = plt.get_backend()
    mng = plt.get_current_fig_manager()
    if plot_backend == 'TkAgg':
        mng.resize(*mng.window.maxsize())
    elif plot_backend == 'wxAgg':
        mng.frame.Maximize(True)
    elif plot_backend == 'Qt4Agg':
        mng.window.showMaximized()

maximize()
plt.show()
Writer = animation.writers['ffmpeg']
writer = Writer(fps=15, metadata=dict(artist='gjkf'), bitrate=1800)
#ani.save("fft.mp4", writer=writer)
