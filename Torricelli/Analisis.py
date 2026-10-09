import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from scipy.optimize import curve_fit

# --- 1. CONFIGURACIÓN Y METROLOGÍA ---
SIGMA_US = 0.3  # Incertidumbre del ultrasónico (cm)
ARCHIVO_CSV = r"C:\Users\ferna\Desktop\torricelli\datos_torricelli.csv"
TIEMPO_CORTE = 130.0  # Segundo donde se corta el goteo final

# --- 2. PARÁMETROS GEOMÉTRICOS DEL SISTEMA FÍSICO ---
H_TANQUE = 17.0            # Altura total del recipiente (cm)
D_TANQUE = 21.0            # Diámetro/Ancho del recipiente (cm)
D_ORIFICIO = 0.6           # Diámetro del orificio de salida (cm)
D_SENSOR_ORIFICIO = 15.5   # Distancia fija del sensor al orificio (cm)
H_AGUA_INICIAL = 6.0       # Altura inicial de la columna de agua (cm)
GRAVEDAD = 981.0           # Aceleración de la gravedad (cm/s^2)

# --- 3. EL MODELO FÍSICO (Parábola de Torricelli) ---
def modelo_torricelli(t, A, B, C):
    return A * t**2 + B * t + C

# --- 4. LECTURA Y CORTE DE DATOS ---
print("Cargando telemetría...")
try:
    df = pd.read_csv(ARCHIVO_CSV)
    t_crudo = df['time_s'].values
    d_crudo = df['distance_cm'].values
    print(f"¡Éxito! {len(t_crudo)} muestras leídas en total.")
except FileNotFoundError:
    print(f"Error: No se encontró el archivo '{ARCHIVO_CSV}'.")
    exit()

mask = t_crudo <= TIEMPO_CORTE
t_arr = t_crudo[mask]
d_arr = d_crudo[mask]

# --- 5. AJUSTE METROLÓGICO Y CÁLCULO TEÓRICO ---
if len(t_arr) > 10:
    u_arr = np.full(len(d_arr), SIGMA_US)
    
    # 5.1 Ajuste Empírico (Mínimos Cuadrados)
    d_inicial_emp = d_arr[0] if len(d_arr) > 0 else 9.5
    popt, pcov = curve_fit(modelo_torricelli, t_arr, d_arr, 
                           p0=[0.001, 0.01, d_inicial_emp], 
                           sigma=u_arr, absolute_sigma=True)
    perr = np.sqrt(np.diag(pcov))
    
    d_fit = modelo_torricelli(t_arr, *popt)
    chi2 = np.sum(((d_arr - d_fit) / u_arr)**2)
    ndof = len(t_arr) - 3  
    chi2_ndof = chi2 / ndof if ndof > 0 else 0

    # 5.2 Cálculo Teórico (Ecuación de Continuidad y Bernoulli)
    Area_tanque = np.pi * (D_TANQUE / 2.0)**2
    Area_orificio = np.pi * (D_ORIFICIO / 2.0)**2
    k_razon = Area_orificio / Area_tanque
    
    A_teo = - (k_razon**2 * GRAVEDAD) / 2.0
    B_teo = k_razon * np.sqrt(2.0 * GRAVEDAD * H_AGUA_INICIAL)
    C_teo = D_SENSOR_ORIFICIO - H_AGUA_INICIAL

    # --- 6. PANEL DE DIAGNÓSTICO INDEPENDIENTE ---
    fig = plt.figure(figsize=(8, 12), layout="tight")
    gs = gridspec.GridSpec(3, 1, height_ratios=[2.5, 1, 1], hspace=0.4)
    
    ax_main = fig.add_subplot(gs[0])
    ax_diff = fig.add_subplot(gs[1])
    ax_res = fig.add_subplot(gs[2])

    # Panel 1: Ajuste
    ax_main.errorbar(t_arr, d_arr, yerr=SIGMA_US, fmt='co', ms=3, ecolor='gray', 
                     capsize=2, alpha=0.8, label=r'Datos Ultrasónico ($2\sigma$)')
    
    t_smooth = np.linspace(min(t_arr), max(t_arr), 200)
    d_smooth = modelo_torricelli(t_smooth, *popt)
    ax_main.plot(t_smooth, d_smooth, 'r-', lw=2.5, 
                 label=f"Ajuste Torricelli\n$\\chi^2/ndof = {chi2_ndof:.2f}$")
    
    ax_main.set_title("Cinemática de Vaciado de Tanque (Ley de Torricelli)")
    ax_main.set_ylabel("Distancia Sensor-Agua $d_{real}$ (cm)")
    ax_main.set_xlabel("Tiempo $t$ (s)")
    ax_main.legend(loc='upper left')

    # --- CAJA DE INFORMACIÓN (ARRIBA A LA DERECHA) ---
    # Toda la info en un solo bloque unificado
    info_text = (
        f"$D_{{tanque}}$ = {D_TANQUE} cm\n"
        f"$D_{{orificio}}$ = {D_ORIFICIO*10} mm\n"
        f"$h_{{inicial}}$ = {H_AGUA_INICIAL} cm\n"
        f"$C_{{teo}} = {C_teo:.2f}$ cm  |  $C_{{fit}} = {popt[2]:.2f}$ cm\n"
        f"$B_{{teo}} = {B_teo:.4f}$  |  $B_{{fit}} = {popt[1]:.4f}$\n"
        f"$A_{{teo}} = {A_teo:.5f}$  |  $A_{{fit}} = {popt[0]:.5f}$"
    )
    # Se coloca en coordenadas relativas (0.95 en X, 0.95 en Y)
    ax_main.text(0.95, 0.95, info_text, transform=ax_main.transAxes, fontsize=10,
                 verticalalignment='top', horizontalalignment='right',
                 bbox=dict(boxstyle='round,pad=0.5', facecolor='white', alpha=0.8, edgecolor='gray'))

    # Panel 2: Diferencia Absoluta
    diff = d_arr - d_fit
    ax_diff.errorbar(t_arr, diff, yerr=SIGMA_US, fmt='co', ms=3, capsize=2, alpha=0.6)
    ax_diff.axhline(0, color='black', lw=1.5)
    ax_diff.set_ylabel("Diferencia\n$d_{real} - d_{fit}$ (cm)")
    ax_diff.set_xlabel("Tiempo $t$ (s)")

    # Panel 3: Residuos Normalizados
    r_i = diff / SIGMA_US
    ax_res.errorbar(t_arr, r_i, yerr=1.0, fmt='co', ms=3, capsize=2, alpha=0.6)
    ax_res.axhline(0, color='black', lw=1.5)
    ax_res.axhline(2, color='red', ls='--', alpha=0.5)
    ax_res.axhline(-2, color='red', ls='--', alpha=0.5)
    ax_res.set_ylabel("Residuos\nnormalizados $r_i$")
    ax_res.set_xlabel("Tiempo $t$ (s)")
    ax_res.set_ylim(-5.5, 5.5)

    # Ajuste exacto de ejes X
    for ax in [ax_main, ax_diff, ax_res]:
        ax.set_xlim(min(t_arr), max(t_arr))
        ax.tick_params(direction='in', right=True, top=True)

    plt.show()
else:
    print("No hay suficientes datos para ajustar.")