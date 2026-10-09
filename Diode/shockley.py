import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import numpy as np
from scipy.optimize import curve_fit
import csv

# --- CONSTANTES FÍSICAS ---
T_kelvin = 300.0      # Temperatura del laboratorio
k_B = 1.380649e-23    # Constante de Boltzmann (J/K)
q_e = 1.60217663e-19  # Carga del electrón (C)
Vt = (k_B * T_kelvin) / q_e  # Voltaje térmico

# --- MODELO IDEAL (Shockley) ---
def modelo_shockley(V, Is_nA, n):
    Is_mA = Is_nA * 1e-6  # Convertir nA a mA para la gráfica
    nVt = n * Vt
    return Is_mA * (np.exp(V / nVt) - 1)

# --- LECTURA DEL ARCHIVO CSV ---
filename = input("Introduce el nombre del archivo csv a leer (ej. datos.csv): ")

V_list = []
I_list = []
uI_list = []

try:
    with open(filename, mode='r') as file:
        reader = csv.reader(file)
        next(reader)  # Saltarse la primera fila (los encabezados)
        for row in reader:
            if row: # Evitar líneas vacías
                V_list.append(float(row[0]))
                I_list.append(float(row[1]))
                uI_list.append(float(row[3])) # Tomamos la incertidumbre de la corriente
    print("¡Datos cargados correctamente!")
except Exception as e:
    print(f"Error al leer el archivo: {e}")
    exit()

V_arr = np.array(V_list)
I_arr = np.array(I_list)
uI_arr = np.array(uI_list)

# --- AJUSTE DE CURVA (Solo 2 parámetros) ---
popt = [1.0, 1.5] 
perr = [0.0, 0.0] 
chi2_ndof = 0.0

print("Calculando ajuste...")
try:
    # OJO: Aquí forzamos el ajuste sobre toda la curva para demostrar que falla en altos voltajes
    popt, pcov = curve_fit(modelo_shockley, V_arr, I_arr, 
                           p0=[1.0, 1.5], 
                           bounds=([1e-6, 1.0], [1e6, 4.0]))
    perr = np.sqrt(np.diag(pcov))
except Exception as e:
    print(f"Advertencia: No se pudo realizar el ajuste óptimo. Error: {e}")

# --- GENERACIÓN DE LA GRÁFICA ---
# Figura más angosta y alta
fig = plt.figure(figsize=(6, 12.5), layout="tight")

# Aumentamos el hspace de 0.05 a 0.35 para que quepan los textos entre las gráficas
gs = gridspec.GridSpec(3, 1, height_ratios=[3, 1, 1], hspace=0.35)

ax_main = fig.add_subplot(gs[0])
ax_ratio = fig.add_subplot(gs[1], sharex=ax_main)
ax_res = fig.add_subplot(gs[2], sharex=ax_main)

V_smooth = np.linspace(0, max(V_arr)*1.05, 200)
I_model_smooth = modelo_shockley(V_smooth, *popt)
I_model_points = modelo_shockley(V_arr, *popt)

# Estadísticos (ndof = N - 2 porque Shockley solo ajusta Is y n)
chi2 = np.sum(((I_arr - I_model_points) / uI_arr)**2)
ndof = len(V_arr) - 2  
chi2_ndof = chi2 / ndof if ndof > 0 else 0

leyenda_modelo = (
    f"Modelo Ideal (Shockley)\n"
    f"$I_s = ({popt[0]:.1f} \\pm {perr[0]:.1f})$ nA\n"
    f"$n = {popt[1]:.2f} \\pm {perr[1]:.2f}$\n"
    f"$\\chi^2/ndof = {chi2_ndof:.2f}$"
)

# PANEL 1: PRINCIPAL
ax_main.plot(V_smooth, I_model_smooth, 'r-', lw=1.8, label=leyenda_modelo, zorder=2)
ax_main.errorbar(V_arr, I_arr, yerr=uI_arr, fmt='mo', markersize=4, 
                 capsize=2, elinewidth=1, label=r'Datos ($2\sigma$)', zorder=1)

ax_main.set_ylabel(r"Corriente $I_D$ (mA)", fontsize=12)
# Agregamos etiqueta X al panel principal
ax_main.set_xlabel(r"Voltaje en el Diodo $V_D$ (V)", fontsize=13)
ax_main.set_xlim(0, max(V_arr)*1.1)
ax_main.set_ylim(-0.5, max(I_arr)*1.15)
ax_main.legend(loc='upper left', fontsize=15, framealpha=0.8, edgecolor='black')
ax_main.tick_params(direction='in', right=True, top=True, labelsize=10)

# PANEL 2: RATIO
I_model_safe = np.where(I_model_points < 1e-6, 1e-6, I_model_points)
ratio = I_arr / I_model_safe
u_ratio = uI_arr / I_model_safe 

ax_ratio.errorbar(V_arr, ratio, yerr=u_ratio, fmt='mo', markersize=3, capsize=2, elinewidth=1)
ax_ratio.axhline(1.0, color='black', linestyle='--', lw=1, zorder=0)
ax_ratio.set_ylabel("Datos/Modelo", fontsize=10)
# Agregamos etiqueta X al panel de ratio
ax_ratio.set_xlabel(r"Voltaje en el Diodo $V_D$ (V)", fontsize=13)
ax_ratio.set_ylim(0, 2)
ax_ratio.tick_params(direction='in', right=True, top=True, labelsize=10)

# PANEL 3: RESIDUOS
r_i = (I_arr - I_model_points) / uI_arr
ax_res.errorbar(V_arr, r_i, yerr=1.0, fmt='mo', markersize=3, capsize=2, elinewidth=1)

ax_res.axhline(0, color='black', lw=1.2, zorder=0)
ax_res.axhline(2.0, color='red', linestyle='--', lw=0.9, alpha=0.5)
ax_res.axhline(-2.0, color='red', linestyle='--', lw=0.9, alpha=0.5)

ax_res.set_ylabel("Residuos\nnormalizados $r_i$", fontsize=10)
ax_res.set_xlabel(r"Voltaje en el Diodo $V_D$ (V)", fontsize=13) 
ax_res.set_ylim(-5, 5)
ax_res.tick_params(direction='in', right=True, top=True, labelsize=10)

# Ya no ocultamos los números del eje X
# plt.setp(ax_main.get_xticklabels(), visible=False)
# plt.setp(ax_ratio.get_xticklabels(), visible=False)

plt.show()