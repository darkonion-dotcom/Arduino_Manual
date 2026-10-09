# Este script procesa N repeticiones, calcula la estadística del ensamble,
# propaga el error instrumental y genera un panel de diagnóstico.

import os
import numpy as np
import pandas as pd
from scipy.optimize import curve_fit
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec

# Definimos el modelo ideal de descarga
def funcion_descarga(t, V0, tau):
    return V0 * np.exp(-t / tau)

# %% [2. PARÁMETROS NOMINALES E INCERTIDUMBRES BASE]
# Aquí definimos la "verdad teórica" y los límites de nuestro hardware

# Resolución del ADC del Arduino (10 bits sobre 5V)
U_INSTRUMENTAL = 5.0 / 1024.0  # ~4.883 mV

# Valores nominales y tolerancias de fábrica
R_NOM = 10000.0        # 10 kOhms
U_R = R_NOM * 0.05     # Tolerancia del 5%
C_NOM = 0.00047        # 470 uF
U_C = C_NOM * 0.20     # Tolerancia del 20%

# Cálculo del Tau teórico y propagación de su error
TAU_TH = R_NOM * C_NOM
U_TAU_TH = np.sqrt((C_NOM * U_R)**2 + (R_NOM * U_C)**2)

print(f"--- Parámetros Teóricos ---")
print(f"Tau Teórico: {TAU_TH:.4f} ± {U_TAU_TH:.4f} s")
print(f"Resolución ADC: {U_INSTRUMENTAL*1000:.3f} mV\n")

# %% [3. ADQUISICIÓN Y CONTROL DE CALIDAD]
# Leemos los 10 archivos CSV generados por el Arduino

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
files = [f"discharge_470uF_10kohms_A{i}.csv" for i in range(1, 11)]

resultados_corridas = []
t_ref, v_ref = None, None  # Guardaremos la corrida 1 como referencia visual

for fname in files:
    path = os.path.join(SCRIPT_DIR, fname)
    if not os.path.exists(path):
        continue
        
    df = pd.read_csv(path)
    t = df['time_s'].values
    v = df['voltage_V'].values
    
    # Guardamos la primera corrida como la muestra representativa para la gráfica
    if fname == "discharge_470uF_10kohms_A1.csv":
        t_ref = t
        v_ref = v
    
    # El peso estadístico (sigma) es el límite de nuestro instrumento
    sigma_v = np.full(len(v), U_INSTRUMENTAL)
    
    # %% [4. AJUSTE INDIVIDUAL POR CORRIDA]
    try:
        # Algoritmo de Levenberg-Marquardt ponderado
        popt, pcov = curve_fit(funcion_descarga, t, v, p0=[5.0, 4.7], 
                               sigma=sigma_v, absolute_sigma=True)
        perr = np.sqrt(np.diag(pcov))
        
        # Bondad de ajuste
        v_fit = funcion_descarga(t, *popt)
        chi2 = np.sum(((v - v_fit) / sigma_v) ** 2)
        ndof = len(t) - 2
        
        resultados_corridas.append({
            'Archivo': fname,
            'V0': popt[0],
            'Tau': popt[1],
            'u_Tau': perr[1],
            'Chi2_ndof': chi2 / ndof if ndof > 0 else 0
        })
    except Exception as e:
        print(f"Error procesando {fname}: {e}")

# %% [5. ESTADÍSTICA DEL ENSAMBLE (GLOBAL)]
# Consolidamos los datos de las 10 repeticiones

df_res = pd.DataFrame(resultados_corridas)
N_corridas = len(df_res)

# Promedios
tau_prom = df_res['Tau'].mean()
v0_prom = df_res['V0'].mean()
chi2_prom = df_res['Chi2_ndof'].mean()

# Incertidumbre total de Tau (Tipo A + Tipo B)
std_tau = df_res['Tau'].std()
u_stat = std_tau / np.sqrt(N_corridas)
u_fit = np.sqrt(np.sum(df_res['u_Tau']**2)) / N_corridas
u_tot = np.sqrt(u_stat**2 + u_fit**2)

# Incertidumbre de V0
v0_err = df_res['V0'].std() / np.sqrt(N_corridas)

print(f"--- Resultados del Ensamble (N={N_corridas}) ---")
print(f"Tau Experimental: {tau_prom:.4f} ± {u_tot:.4f} s")
print(f"V0 Experimental: {v0_prom:.4f} ± {v0_err:.4f} V")
print(f"Chi2/ndof Promedio: {chi2_prom:.2f}\n")

# %% [6. GENERACIÓN DE DIAGNÓSTICO VISUAL]
# Creamos la figura de 3 paneles para evidenciar el límite del ADC

if t_ref is not None:
    plt.rcParams['text.usetex'] = False 
    fig = plt.figure(figsize=(8, 13))
    gs = gridspec.GridSpec(3, 1, height_ratios=[2, 1, 1], hspace=0.35)
    
    ax_main = fig.add_subplot(gs[0])
    ax_diff = fig.add_subplot(gs[1])
    ax_res = fig.add_subplot(gs[2])
    
    # Modelo ideal con los parámetros del ensamble global
    t_smooth = np.linspace(0, max(t_ref) * 1.05, 500)
    V_model_smooth = funcion_descarga(t_smooth, v0_prom, tau_prom)
    V_model_points = funcion_descarga(t_ref, v0_prom, tau_prom)
    
    # Panel 1: Ajuste Macroscópico
    ax_main.errorbar(t_ref, v_ref, yerr=U_INSTRUMENTAL, fmt='o', markersize=3, 
                     color='navy', ecolor='dimgray', capsize=2, alpha=0.5, 
                     label='Datos Experimentales (Muestra Representativa)')
    ax_main.plot(t_smooth, V_model_smooth, color='firebrick', lw=2.5, alpha=0.9, 
                 label='Modelo Ideal Ensamble: $V_0 e^{-t/\\tau}$')
    
    ax_main.set_ylabel("Voltaje (V)", fontsize=14)
    ax_main.set_xlabel("Tiempo (s)", fontsize=14)
    ax_main.set_xlim(-1, max(t_ref) * 1.05)
    ax_main.set_ylim(-0.2, 5.5)
    
    texto_parametros = (
        f"Resultados Ensamble (N={N_corridas}):\n"
        f"τ_exp = ({tau_prom:.4f} ± {u_tot:.4f}) s\n"
        f"χ²/ndof prom = {chi2_prom:.2f}"
    )
    props = dict(boxstyle='round,pad=0.8', facecolor='white', alpha=0.95, edgecolor='gray')
    ax_main.text(0.48, 0.95, texto_parametros, transform=ax_main.transAxes, fontsize=11,
                 verticalalignment='top', bbox=props)
    ax_main.legend(loc='lower left', fontsize=12)
    
    # Panel 2: Diferencia Absoluta
    diff_exp_fit = (v_ref - V_model_points)
    ax_diff.errorbar(t_ref, diff_exp_fit, yerr=U_INSTRUMENTAL, fmt='o', markersize=3, 
                     color='navy', ecolor='dimgray', capsize=2, alpha=0.6)
    ax_diff.axhline(0, color='black', lw=1.5)
    ax_diff.set_ylabel("Diferencia\n$V_{exp} - V_{fit}$ (V)", fontsize=12)
    ax_diff.set_xlabel("Tiempo (s)", fontsize=14)
    ax_diff.set_xlim(-1, max(t_ref) * 1.05)
    ax_diff.set_ylim(-0.15, 0.15)
    
    # Panel 3: Residuos Normalizados (Evidencia del ADC)
    r_i = diff_exp_fit / U_INSTRUMENTAL
    ax_res.errorbar(t_ref, r_i, yerr=1.0, fmt='o', markersize=3, color='navy', 
                    ecolor='dimgray', capsize=2, alpha=0.6)
    ax_res.axhline(0, color='black', lw=1.5)
    ax_res.axhline(2.0, color='firebrick', linestyle='--', lw=1.2)
    ax_res.axhline(-2.0, color='firebrick', linestyle='--', lw=1.2)
    ax_res.set_ylabel("Residuos\nnormalizados $r_i$", fontsize=12)
    ax_res.set_xlabel("Tiempo (s)", fontsize=14)
    ax_res.set_xlim(-1, max(t_ref) * 1.05)
    ax_res.set_ylim(-4.5, 4.5)   
    plt.subplots_adjust(bottom=0.08)
    plt.show()