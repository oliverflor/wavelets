import os
import pickle
from time import time
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
from matplotlib.gridspec import GridSpec
import numpy as np
from waveletFunctions import wave_signif, wavelet

def gaussian(x, *params):
    y = np.zeros_like(x)
    for i in range(0, len(params), 3):
        ctr = params[i]
        amp = params[i+1]
        wid = params[i+2] #sigma
        y = y + np.abs(amp) * np.exp(-((x - ctr)/wid) ** 2)
    return y

def lorentzian(x, *params):
    y = np.zeros_like(x)
    for i in range(0, len(params), 3):
        ctr = params[i]
        amp = params[i + 1]
        wid = params[i + 2]
        y += np.abs(amp) * (wid ** 2) / ((x - ctr) ** 2 + wid ** 2)
    return y

def asymmetric_gaussian(x, *params):
    y = np.zeros_like(x)
    for i in range(0, len(params), 4):
        ctr = params[i]
        amp = params[i + 1]
        wl = params[i + 2]
        wr = params[i + 3]
        y += np.abs(amp) * np.where(x < ctr,
                                    np.exp(-((x - ctr) / wl) ** 2),
                                    np.exp(-((x - ctr) / wr) ** 2))
    return y

def decompose_components(time, popt, model_name):
    """
    Given the fitted parameters and model name,
    return individual component fits for plotting.
    """
    if model_name == 'gaussian':
        func = gaussian
        n_params = 3
    elif model_name == 'lorentzian':
        func = lorentzian
        n_params = 3
    elif model_name == 'asymmetric_gaussian':
        func = asymmetric_gaussian
        n_params = 4
    else:
        raise ValueError(f"Unknown model: {model_name}")

    n_components = len(popt) // n_params
    fit_components = np.zeros((n_components, len(time)))

    for i in range(n_components):
        params = popt[i * n_params:(i + 1) * n_params]
        fit_components[i] = func(time, *params)
    return fit_components

def analyse_series(string, foldername, 
                   time, fit, counts, slope, model, popt,
                   afino_status, afino_period,
                   error=None):
    
    file = string.replace('-', '')
    base = os.path.dirname(foldername)
    dir = os.path.join(base, 'save')
    savedir = os.path.join(base, 'wavelet')
    # os.makedirs(os.path.expanduser(savedir), exist_ok=True)
    filepath = os.path.join(dir, 'variables_' + file + '.pkl')

    # with open(filepath, 'rb') as f:
    #     data = pickle.load(f)
    

    # time = data['time']
    # counts = data['counts']
    # fit = data['fit']
    # slope = data['mean_slope']
    # stderr = data['std_slope']
    # if afino:
    #     afino_model = data['afino_best_model']
    #     afino_period = data['afino_best_period'][0]
    #     if data['afino_qpp_detected'][0] == 'True':
    #         status = "QPP detected"
    #         S = data['afino_qpp_detected'][1]
    #     else:
    #         status = "No QPP detected"


    if error:
        wavelet_analysis(filepath, savedir, file,
                         model, time, fit, counts, slope, popt,
                         afino_status, afino_period,
                         stderr=error)
    else:
        wavelet_analysis(filepath, savedir, file,
                         model, time, fit, counts, slope, popt,
                         afino_status, afino_period)

def wavelet_analysis(filepath, savedir, file, 
                     model_name, time, fit, counts, slope, popt,
                     afino, afino_period,
                     stderr=None):

    # with open(filepath, 'rb') as f:
    #     data = pickle.load(f)
    # model_name = data['model_name']
    # fit = data['fit']
    # fit_para = data['fit_para']

    fits = decompose_components(time, popt, model_name)


    sst = counts - np.mean(counts) 
    variance = np.std(sst, ddof=1) ** 2

    # ----------C-O-M-P-U-T-A-T-I-O-N------S-T-A-R-T-S------H-E-R-E---------------
    if 0:
        variance = 1.0
        sst = sst / np.std(sst, ddof=1)
    n = len(sst)
    dt = 1
    pad = 5  # pad the time series with zeroes (recommended)
    dj = 0.1  # this will do 4 sub-octaves per octave
    s0 = 3 * dt  # this says start at a scale of 6 months
    j1 = 7 / dj  # this says do 7 powers-of-two with dj sub-octaves each
    lag1 = 0.72  # lag-1 autocorrelation for red noise background
    print("lag1 = ", lag1)
    mother = 'MORLET'

    # Wavelet transform:
    wave, period, scale, coi = wavelet(sst, dt, pad, dj, s0, j1, mother)
    power = (np.abs(wave)) ** 2  # compute wavelet power spectrum
    global_ws = (np.sum(power, axis=1) / n)  # time-average over all times

    # Significance levels:
    signif = wave_signif(([variance]), dt=dt, sigtest=0, scale=scale,  siglvl =0.95,  # default siglvl =0.95,
        lag1=lag1, mother=mother)
    # expand signif --> (J+1)x(N) array
    sig95 = signif[:, np.newaxis].dot(np.ones(n)[np.newaxis, :])
    sig95 = power / sig95  # where ratio > 1, power is significant

    # Global wavelet spectrum & significance levels:
    dof = n - scale  # the -scale corrects for padding at edges
    global_signif = wave_signif(variance, dt=dt, scale=scale, sigtest=1, siglvl =0.95,  # default siglvl =0.95,
        lag1=lag1, dof=dof, mother=mother)

    # --- Plot time series
    fig = plt.figure(figsize=(12, 8))
    gs = GridSpec(2, 3, hspace=0.3, wspace=0.2)
    plt.subplots_adjust(left=0.1, bottom=0.05, right=0.9, top=0.95,
                    wspace=0, hspace=0)

    plt.subplot(gs[0, 0:2])
    plt.plot(time, fit, label=f'linear combination of {model_name} components')
    plt.plot(time, counts, label='input data', linewidth=0.5)
    for i in range(len(fits)):
        plt.plot(time, fits[i])
    plt.xlabel('Time')
    plt.ylabel('Count Rate')
    plt.title('a) STIX Time Series (Decomposed)')
    plt.legend()

    # --- Contour plot wavelet power spectrum

    plt3 = plt.subplot(gs[1, 0:2])

    # *** or use 'contour'
    CS = plt.contourf(time, period, power, levels=255, cmap="Blues")      # viridis    Blues
    im = plt.contourf(CS)

    plt.xlabel('Time')
    plt.ylabel('Period (s)')
    plt.title('b) Wavelet Power Spectrum')

    # 95% significance contour, levels at -99 (fake) and 1 (95# signif)
    plt.contour(time, period, sig95, [-99, 1], colors='k', linestyles = ':')
    # cone-of-influence, anything "below" is dubious
    plt.fill_between(time, coi * 0 + period[-1], coi, facecolor="none",
        edgecolor="#00000040", hatch='x', linewidth=0.5)
    #plotting
    plt.plot(time, coi, 'k', linewidth=1)
    # format y-scale
    plt3.set_yscale('log', base=2, subs=None)
    plt.ylim([np.min(period), np.max(period)])
    ax = plt.gca().yaxis
    ax.set_major_formatter(ticker.ScalarFormatter())
    plt3.ticklabel_format(axis='y', style='plain')
    plt3.invert_yaxis()

    # --- Plot global wavelet spectrum
    plt4 = plt.subplot(gs[1, -1])
    plt.plot(global_ws/np.max(global_ws), period)
    plt.plot(global_signif/np.max(global_ws), period, '--')
    plt.xlabel('Power')
    plt.title('c) Global Wavelet Spectrum')
    plt.xlim([0, 1.09])

    # format y-scale
    plt4.set_yscale('log', base=2, subs=None)
    plt4.axhline(y=slope, color='red', label=f"Decomposition: f={slope:.3f} s = {1/slope:.3f} Hz")
    if afino:
        plt4.axhline(y=afino_period, color='green',
                    label=f"Afino (QPP detected): f={afino_period:.3f} s = {1/afino_period:.3f} Hz")
    else:
        plt4.axhline(y=afino_period, color='green', linestyle='--',
                    label=f"Afino: f={afino_period:.3f} s = {1/afino_period:.3f} Hz")
        
    if stderr is not None:
        slope_min = slope + stderr
        slope_max = slope - stderr 
        plt4.axhspan(slope_min, slope_max, alpha=0.3, color='red')
    plt4.legend()
    plt.ylim([np.min(period), np.max(period)])

    qpp_periods = []
    signif_level = []
    wavelet_power = []
    for i in range(len(global_ws/np.max(global_ws))):
        qpp_periods.append(period[i])
        signif_level.append(global_signif[i]/np.max(global_ws))
        wavelet_power.append(global_ws[i]/np.max(global_ws))

    # data.update({
    # # Global (time-averaged) info
    # "wavelet_qpp_periods": period.tolist(),
    # "wavelet_global_signif": global_signif.tolist(),
    # "wavelet_global_power": global_ws.tolist(),

    # # Time-resolved info
    # "wavelet_power_matrix": power.tolist(),   # shape: (len(period), len(time))
    # "wavelet_coi": coi.tolist(),              # shape: (len(time),)
    # "wavelet_local_signif": sig95.tolist(),   # shape: (len(period), len(time))
    # "wavelet_time": time.tolist()             # store time axis for later
    # })
    # with open(filepath, 'wb') as f:
    #     pickle.dump(data, f)


    ax = plt.gca().yaxis
    ax.set_major_formatter(ticker.ScalarFormatter())
    plt4.ticklabel_format(axis='y', style='plain')
    plt4.invert_yaxis()
    # plt.savefig(os.path.join(savedir, f'wavelet_analysis_{file}.png'))
    plt.show()
    
