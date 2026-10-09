import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import numpy as np
from scipy.optimize import curve_fit
from scipy.special import lambertw
import csv

# --- CONSTANTES FÍSICAS ---
T_kelvin = 300.0      # Temperatura del laboratorio
k_B = 1.380649e-23    # Constante de Boltzmann (J/K)
q_e = 1.60217663e-19  # Carga del electrón (C)
Vt = (k_B * T_kelvin) / q_e  # Voltaje térmico

# --- MODELO DEL DIODO REAL (W-Lambert) ---
def modelo_diodo_real(V, Is_nA, n, Rs):
    Is_A = Is_nA * 1e-9  
    nVt = n * Vt
    arg = (Is_A * Rs / nVt) * np.exp((V + Is_A * Rs) / nVt)
    W = lambertw(arg).real
    I_A = (nVt / Rs) * W - Is_A
    return I_A * 1000.0  # Retorna en mA

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

# --- AJUSTE DE CURVA ---
popt = [1.0, 1.5, 5.0] 
perr = [0.0, 0.0, 0.0] 
chi2_ndof = 0.0

print("Calculando ajuste...")
try:
    popt, pcov = curve_fit(modelo_diodo_real, V_arr, I_arr, 
                           p0=[1.0, 1.5, 5.0], 
                           bounds=([1e-6, 1.0, 0.01], [1e6, 4.0, 500.0]))
    perr = np.sqrt(np.diag(pcov))
except Exception as e:
    print(f"Advertencia: No se pudo realizar el ajuste óptimo. Error: {e}")

# --- GENERACIÓN DE LA GRÁFICA ---
# Ajuste de tamaño: Más angosto (6) y más alto (12.5)
fig = plt.figure(figsize=(6, 12.5), layout="tight")
# Aumentamos el espacio vertical para que no se encimen las etiquetas al recortar
gs = gridspec.GridSpec(3, 1, height_ratios=[3, 1, 1], hspace=0.35)

ax_main = fig.add_subplot(gs[0])
ax_ratio = fig.add_subplot(gs[1], sharex=ax_main)
ax_res = fig.add_subplot(gs[2], sharex=ax_main)

# Margen reducido al 2% (1.02) para que la línea roja termine pegada al borde
V_smooth = np.linspace(0, max(V_arr)*1.02, 200)
I_model_smooth = modelo_diodo_real(V_smooth, *popt)
I_model_points = modelo_diodo_real(V_arr, *popt)

# Estadísticos
chi2 = np.sum(((I_arr - I_model_points) / uI_arr)**2)
ndof = len(V_arr) - 3  
chi2_ndof = chi2 / ndof if ndof > 0 else 0

leyenda_modelo = (
    f"Modelo Real (W-Lambert)\n"
    f"$I_s = ({popt[0]:.1f} \\pm {perr[0]:.1f})$ nA\n"
    f"$n = {popt[1]:.2f} \\pm {perr[1]:.2f}$\n"
    f"$R_s = ({popt[2]:.1f} \\pm {perr[2]:.1f}) \\ \\Omega$\n"
    f"$\\chi^2/ndof = {chi2_ndof:.2f}$"
)

# PANEL 1: PRINCIPAL
ax_main.plot(V_smooth, I_model_smooth, 'r-', lw=1.8, label=leyenda_modelo, zorder=2)
ax_main.errorbar(V_arr, I_arr, yerr=uI_arr, fmt='mo', markersize=4, 
                 capsize=2, elinewidth=1, label=r'Datos ($2\sigma$)', zorder=1)

ax_main.set_ylabel(r"Corriente $I_D$ (mA)", fontsize=12)
# Etiqueta X agregada para poder recortar la imagen sola
ax_main.set_xlabel(r"Voltaje en el Diodo $V_D$ (V)", fontsize=13)

# Márgenes de los ejes reducidos (1.02 en X y 1.05 en Y) para apretar la gráfica a la derecha
ax_main.set_xlim(0, max(V_arr)*1.02)
ax_main.set_ylim(-0.5, max(I_arr)*1.05)

# Tamaño de fuente aumentado en la leyenda
ax_main.legend(loc='upper left', fontsize=13, framealpha=0.8, edgecolor='black')
ax_main.tick_params(direction='in', right=True, top=True, labelsize=10)

# PANEL 2: RATIO
I_model_safe = np.where(I_model_points < 1e-6, 1e-6, I_model_points)
ratio = I_arr / I_model_safe
u_ratio = uI_arr / I_model_safe 

ax_ratio.errorbar(V_arr, ratio, yerr=u_ratio, fmt='mo', markersize=3, capsize=2, elinewidth=1)
ax_ratio.axhline(1.0, color='black', linestyle='--', lw=1, zorder=0)
ax_ratio.set_ylabel("Datos/Modelo", fontsize=10)
# Etiqueta X agregada
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

# Las líneas que ocultaban los números en X fueron eliminadas intencionalmente
plt.show()