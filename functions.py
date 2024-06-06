###########################################################
#   Author: Kelan Solomon                                 #
#   Create date: 02/05/2024                               #
#   Last change date: 02/05/2024                          #
#   Changed by: KS                                        #
#   Comment:                                              #
###########################################################


import numpy as np
import SeekCamera
import time
import pandas as pd
import seaborn as sn
import matplotlib.pyplot as plt 
from scipy.interpolate import interp1d
from scipy.integrate import solve_ivp
import os

# Get all slices of the df 
def get_slice(df,r): 
    slice = []
    for frame in df.img:
        slice.append(frame[r])
    return slice

#  Get new data from camera (-> Need camera connection) and pickle it
def get_new_data(runtime,name):
    # Get data
    times = SeekCamera.get_data(runtime)
    times = np.array(times)
    time.sleep(2)

    df = pd.read_csv("./thermography-E5D608D31715.csv",sep=" ")
    df.to_csv('D:/Data_RAW/{}_raw.csv'.format(name))
    # Seperate into frames
    l_frames = []
    for i in range(0, len(df), 240):
        temperature_data = df.iloc[i:i+239, :].values.astype(float)
        l_frames.append(temperature_data)
    # Create DF
    frames = pd.DataFrame({'img': l_frames})
    frames["time_stamp"] = times
    frames["DT"] = ((times-times[0])/(10**9)).round(2)

    #Pickle
    frames.to_pickle("D:/Data_PKL/{}".format(name))


def first_n_last(frames,ax1,ax2):
    F=sn.heatmap(frames.img[0], ax=ax1)
    L=sn.heatmap(frames.img[len(frames)-1], ax=ax2)
    F.set_title("First frame")
    L.set_title("Last frame")
    return F,L

def slice_analysis(frames, slice, ax1,ax2,interp_time=0.1):
    interp_times = np.arange(0, frames.DT.max(), interp_time)
    interp_temps = []
    for _, frame in enumerate(np.transpose(slice)):
        interp_func = interp1d(frames.DT.tolist(), frame)
        new_temps = interp_func(interp_times)
        interp_temps.append(new_temps)

    # Create DataFrame
    df_temps = pd.DataFrame(interp_temps, columns=interp_times)

    # Plot heatmaps
    L = sn.heatmap(df_temps, ax= ax1)

    gradient = df_temps.diff(axis=1)/interp_time
    R = sn.heatmap(gradient, ax=ax2)


    # Adjust plot
    for ax in [ax1,ax2]:
        num_ticks = 5  # Change this number to adjust the number of ticks
        ax.set_xticks(np.linspace(0, len(interp_times) - 1, num_ticks, dtype=int))
        x_tick_positions = np.linspace(0, len(interp_times) - 1, num_ticks, dtype=int)
        ax.set_xticklabels([f"{interp_times[i]:.0f}" for i in x_tick_positions])
    L.set_title("Middle slice over time [deg]")
    L.set_ylabel("Pixel")
    L.set_xlabel("Time [s]")
    L.set_label('Temperature [°C]')
    R.set_title("Middle slice time gradient [°C/s]")
    R.set_xlabel("Time [s]")

def global_analyse(path, interp_time = 0.1):
    frames = pd.read_pickle("{}".format(path))
    _, axs = plt.subplots(3,2,figsize= (12,16), sharex= False, sharey=False)
    axs = axs.ravel()

    
    # First and last frame
    first_n_last(frames, axs[0], axs[1])

    # Create time series of Smiddle slice 
    slice = get_slice(frames, 120)
    slice_analysis(frames, slice, axs[2],axs[3])
    # Interpolation of time series
    

    sn.lineplot(slice[-1], ax=axs[4])
    sn.lineplot(y=np.transpose(slice)[int(len(np.transpose(slice))/2)], x=frames.DT, ax= axs[5])

    axs[4].set_title("Temperature at last frame [{}s]".format(frames.DT.max()))
    axs[4].set_ylabel("Temperature [°C]")

    axs[5].set_title("Middle pixle heat over time")
    axs[5].set_ylabel("Temperature [°C]")


    plt.tight_layout()
    plt.show()

def model_predict(start, end, PWM, T0=298):
    U = 10 # V
    alpha = 0.00586
    m = 6.25*10**(-5)
    c = 880
    h = 25
    S = 5e-3*5e-3
    R_0 = 6
    T_R0 = 298
    T_0 = 298
    Ri = 4.8 # Internal Power suplly resistance[ohms]
    KR = 0.4
    KL = 5

    omega = 5.67e-8
    emi = 0.75

    def dT_dt(t,T):
        return(PWM*U**2*(R_0*(1+KR*alpha*(T-T_R0)))/(Ri + R_0*(1+KR*alpha*(T-T_R0)))**2- KL*(h*S*(T-T_0)+omega*emi*S*(T**4-T_0**4)))/((m*c))

    t_span = (start,end)
    t_eval = np.linspace(start,end,1600)

    y0 = [T0]
    sol = solve_ivp(dT_dt, t_span, y0, t_eval= t_eval)

    return(sol.y[0,:], t_eval)


def plot_F(filepath, title):
    for f in os.listdir(filepath):
        frames = pd.read_pickle("{}{}".format(filepath,f))
        slice = get_slice(frames, 120)
        sn.lineplot(y=np.transpose(slice)[int(len(np.transpose(slice))/2)], x=frames.DT, label = f)
    plt.xlabel("time [s]")
    plt.ylabel("T[°C]")
    plt.title(title)
    plt.tight_layout()
    # plt.show()

def plot_F_corrected(filepath, title):
    for f in os.listdir(filepath):
        frames = pd.read_pickle("{}{}".format(filepath,f))
        slice = get_slice(frames, 120)
        sn.lineplot(y=np.transpose(slice)[int(len(np.transpose(slice))/2)]*0.97/0.75, x=frames.DT, label = f)
    plt.xlabel("time [s]")
    plt.ylabel("T[°C]")
    plt.title(title)
    plt.tight_layout()

