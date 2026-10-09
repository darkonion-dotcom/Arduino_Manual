#Analisis
import os
import numpy as np
import pandas as pd
from scipy.optimize import curve_fit
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec

def funcion_descarga(t, V0, tau):
    return V0 * np.exp(-t / tau)

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = SCRIPT_DIR
U_INSTRUMENTAL = 5.0 / 1023.0

R_NOM = 10000.0
U_R = R_NOM * 0.05
C_NOM = 0.00047
U_C = C_NOM * 0.20

TAU_TH = R_NOM * C_NOM
U_TAU_TH = np.sqrt((C_NOM * U_R)**2 + (R_NOM * U_C)**2)

files = [f"discharge_470uF_10kohms_A{i}.csv" for i in range(1, 11)]

resultados = []
t_all, v_all = None, None

for fname in files:
    path = os.path.join(DATA_DIR, fname)
    if not os.path.exists(path):
        continue
        
    df = pd.read_csv(path)
    t = df['time_s'].values
    v = df['voltage_V'].values
    
    if fname == "discharge_470uF_10kohms_A1.csv":
        t_all = t
        v_all = v
    
    sigma_v = np.full(len(v), U_INSTRUMENTAL)
    
    try:
        popt, pcov = curve_fit(funcion_descarga, t, v, p0=[5.0, 4.7], sigma=sigma_v, absolute_sigma=True)
        perr = np.sqrt(np.diag(pcov))
        
        v_fit = funcion_descarga(t, *popt)
        chi2 = np.sum(((v - v_fit) / sigma_v) ** 2)
        ndof = len(t) - 2
        
        resultados.append({
            'Archivo': fname,
            'V0': popt[0],
            'Tau': popt[1],
            'u_Tau': perr[1],
            'Chi2_ndof': chi2 / ndof if ndof > 0 else 0
        })
    except:
        pass

df_res = pd.DataFrame(resultados)
tau_prom = df_res['Tau'].mean()
std_tau = df_res['Tau'].std()
u_stat = std_tau / np.sqrt(len(df_res))
u_fit = np.sqrt(np.sum(df_res['u_Tau']**2)) / len(df_res)
u_tot = np.sqrt(u_stat**2 + u_fit**2)
v0_prom = df_res['V0'].mean()
v0_err = df_res['V0'].std() / np.sqrt(len(df_res))
chi2_prom = df_res['Chi2_ndof'].mean()

if t_all is not None:
    sigma_v_all = np.full(len(v_all), U_INSTRUMENTAL)
    popt_all, _ = curve_fit(funcion_descarga, t_all, v_all, p0=[5.0, 4.7], sigma=sigma_v_all, absolute_sigma=True)
    
    plt.rcParams['text.usetex'] = False 
    fig = plt.figure(figsize=(8, 13))
    gs = gridspec.GridSpec(3, 1, height_ratios=[2, 1, 1], hspace=0.35)
    
    ax_main = fig.add_subplot(gs[0])
    ax_diff = fig.add_subplot(gs[1])
    ax_res = fig.add_subplot(gs[2])
    
    t_smooth = np.linspace(0, max(t_all) * 1.05, 500)
    V_model_smooth = funcion_descarga(t_smooth, *popt_all)
    
    ax_main.errorbar(t_all, v_all, yerr=U_INSTRUMENTAL, fmt='o', markersize=3, color='navy', ecolor='dimgray', capsize=2, elinewidth=1, markeredgecolor='navy', alpha=0.5, label='Datos Experimentales (A1)', zorder=1)
    ax_main.plot(t_smooth, V_model_smooth, color='firebrick', linestyle='-', lw=2.5, alpha=0.9, label='Modelo Ideal: V(t) = V0 e^(-t/τ)', zorder=2)
    
    ax_main.set_ylabel("Voltaje (V)", fontsize=14)
    ax_main.set_xlabel("Tiempo (s)", fontsize=14)
    ax_main.set_xlim(-1, max(t_all) * 1.05)
    ax_main.set_ylim(-0.2, 5.5)
    ax_main.tick_params(direction='in', right=True, top=True, labelsize=12)
    
    texto_parametros = (
        "Parámetros Nominales:\n"
        f"R = ({R_NOM/1000:.1f} ± {U_R/1000:.1f}) kΩ\n"
        f"C = ({C_NOM*1e6:.0f} ± {U_C*1e6:.0f}) µF\n"
        f"τ_teórico = ({TAU_TH:.2f} ± {U_TAU_TH:.2f}) s\n\n"
        "Resultados del Ensamble (N=10):\n"
        f"τ_exp = ({tau_prom:.4f} ± {u_tot:.4f}) s\n"
        f"V0_exp = ({v0_prom:.4f} ± {v0_err:.4f}) V\n"
        f"χ²/ndof prom = {chi2_prom:.2f}"
    )
    
    props = dict(boxstyle='round,pad=0.8', facecolor='white', alpha=0.95, edgecolor='gray')
    ax_main.text(0.48, 0.95, texto_parametros, transform=ax_main.transAxes, fontsize=11,
                 verticalalignment='top', bbox=props, zorder=3)
    
    ax_main.legend(loc='lower left', fontsize=12, framealpha=0.95, edgecolor='gray')
    
    V_model_points = funcion_descarga(t_all, *popt_all)
    diff_exp_fit = (v_all - V_model_points)
    
    ax_diff.errorbar(t_all, diff_exp_fit, yerr=U_INSTRUMENTAL, fmt='o', markersize=3, color='navy', ecolor='dimgray', capsize=2, elinewidth=1, markeredgecolor='navy', alpha=0.6)
    ax_diff.axhline(0, color='black', lw=1.5, zorder=2)
    
    ax_diff.set_ylabel("Diferencia\n$V_{exp} - V_{fit}$ (V)", fontsize=12)
    ax_diff.set_xlabel("Tiempo (s)", fontsize=14)
    ax_diff.set_xlim(-1, max(t_all) * 1.05)
    ax_diff.set_ylim(-0.15, 0.15)
    ax_diff.tick_params(direction='in', right=True, top=True, labelsize=12)
    
    r_i = diff_exp_fit / U_INSTRUMENTAL
    
    ax_res.errorbar(t_all, r_i, yerr=1.0, fmt='o', markersize=3, color='navy', ecolor='dimgray', capsize=2, elinewidth=1, markeredgecolor='navy', alpha=0.6)
    ax_res.axhline(0, color='black', lw=1.5, zorder=2)
    ax_res.axhline(2.0, color='firebrick', linestyle='--', lw=1.2, alpha=0.8)
    ax_res.axhline(-2.0, color='firebrick', linestyle='--', lw=1.2, alpha=0.8)
    
    ax_res.set_ylabel("Residuos\nnormalizados $r_i$", fontsize=12)
    ax_res.set_xlabel("Tiempo (s)", fontsize=14)
    ax_res.set_xlim(-1, max(t_all) * 1.05)
    ax_res.set_ylim(-4.5, 4.5)
    ax_res.tick_params(direction='in', right=True, top=True, labelsize=12)
    
    plt.subplots_adjust(bottom=0.08)
    plt.show()