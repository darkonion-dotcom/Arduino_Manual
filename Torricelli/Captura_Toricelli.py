import serial
import time
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit
import os

# --- 1. CONFIGURACIÓN DEL EXPERIMENTO ---
PUERTO = 'COM4'          # Puerto en Windows
BAUD_RATE = 9600
TIEMPO_VACIADO = 160.0   # Cálculo hidrodinámico: ~135s para vaciar 6cm en un tanque de 21cm
SIGMA_US = 0.3           # Incertidumbre del ultrasónico (cm)
UMBRAL_GATILLO = 0.2     # Diferencia en cm (2 mm) para disparar el inicio

# --- 2. EL MODELO FÍSICO (Parábola de Torricelli) ---
def modelo_torricelli(t, A, B, C):
    return A * t**2 + B * t + C

# --- 3. CAPTURA DE DATOS VÍA SERIAL (CON GATILLO AUTOMÁTICO) ---
tiempos = []
distancias = []

print("Conectando al microcontrolador...")
try:
    with serial.Serial(PUERTO, BAUD_RATE, timeout=1) as ser:
        print("\n[FASE 1] Calculando línea base de distancia (NO quite el tapón)...")
        baseline_reads = []
        
        while len(baseline_reads) < 20:
            if ser.in_waiting > 0:
                linea = ser.readline().decode('utf-8').strip()
                try:
                    t_val, d_val = map(float, linea.split(','))
                    if 8.0 < d_val < 16.5:
                        baseline_reads.append(d_val)
                except ValueError:
                    pass
                    
        d_inicial_promedio = np.mean(baseline_reads)
        print(f"Línea base fijada en: {d_inicial_promedio:.2f} cm")
        
        input("\nPresione ENTER y quite el tapón en cualquier momento...")
        print("Esperando caída de agua para disparar cronómetro...")
        
        triggered = False
        start_time = 0
        
        while True:
            if ser.in_waiting > 0:
                linea = ser.readline().decode('utf-8').strip()
                try:
                    t_val, d_val = map(float, linea.split(','))
                    
                    if 8.0 < d_val < 16.5:
                        if not triggered:
                            if d_val > (d_inicial_promedio + UMBRAL_GATILLO):
                                triggered = True
                                start_time = time.time()
                                print(f"¡GATILLO ACTIVADO! Grabando {TIEMPO_VACIADO} segundos...")
                                tiempos.append(0.0)
                                distancias.append(d_val)
                        else:
                            t_actual = time.time() - start_time
                            if t_actual <= TIEMPO_VACIADO:
                                tiempos.append(t_actual)
                                distancias.append(d_val)
                            else:
                                print("\n¡Tiempo de grabación finalizado!")
                                break 
                except ValueError:
                    pass
except Exception as e:
    print(f"Error de conexión: {e}. Verifique el puerto COM4.")

# --- 4. GUARDADO DE DATOS CRUDOS Y ANÁLISIS ---
if len(tiempos) > 10:
    # Guardar en CSV inmediatamente
    df = pd.DataFrame({'time_s': tiempos, 'distance_cm': distancias})
    directorio_actual = os.path.dirname(os.path.abspath(__file__)) if '__file__' in locals() else os.getcwd()
    archivo_csv = os.path.join(directorio_actual, "datos_torricelli.csv")
    df.to_csv(archivo_csv, index=False)
    print(f"\n[+] Datos crudos guardados exitosamente en: {archivo_csv}")

    print("\nProcesando el ajuste no lineal...")
    t_arr = np.array(tiempos)
    d_arr = np.array(distancias)
    
    # 1. RECORTES DE TRANSITORIOS (Data Trimming)
    # Ajustamos la máscara para ignorar el oleaje inicial (< 20s) y el goteo final (> 130s)
    mask = (t_arr > 20.0) & (t_arr < 130.0)
    
    t_clean = t_arr[mask]
    d_clean = d_arr[mask]
    u_clean = np.full(len(d_clean), SIGMA_US)
    
    if len(t_clean) > 10:
        try:
            # Ajustamos solo la parte limpia
            popt, pcov = curve_fit(modelo_torricelli, t_clean, d_clean, 
                                   p0=[0.001, 0.01, d_inicial_promedio], 
                                   sigma=u_clean, absolute_sigma=True)
            perr = np.sqrt(np.diag(pcov))
            
            d_fit = modelo_torricelli(t_clean, *popt)
            chi2 = np.sum(((d_clean - d_fit) / u_clean)**2)
            ndof = len(t_clean) - 3  
            chi2_ndof = chi2 / ndof if ndof > 0 else 0

            print("\n--- EXTRACCIÓN PARAMÉTRICA ---")
            print(f"Distancia inicial (C): {popt[2]:.2f} ± {perr[2]:.2f} cm")
            print(f"Término lineal (B):    {popt[1]:.4f} ± {perr[1]:.4f}")
            print(f"Término cuadrático (A):{popt[0]:.4f} ± {perr[0]:.4f}")
            print(f"Chi2/ndof:             {chi2_ndof:.2f}")

            # 2. GRÁFICA COMPARATIVA
            plt.figure(figsize=(8, 6), layout="tight")
            
            # Ploteamos los datos "Sucios" en gris de fondo
            plt.errorbar(t_arr, d_arr, yerr=SIGMA_US, fmt='.', color='lightgray', ms=2, 
                         alpha=0.4, label='Datos descartados (Oleaje/Goteo)')
            
            # Ploteamos los datos "Limpios" en cyan
            plt.errorbar(t_clean, d_clean, yerr=SIGMA_US, fmt='co', ms=3, ecolor='gray', 
                         capsize=2, alpha=0.8, label=r'Régimen Continuo ($2\sigma$)')
            
            # Curva de ajuste
            t_smooth = np.linspace(min(t_clean), max(t_clean), 200)
            d_smooth = modelo_torricelli(t_smooth, *popt)
            plt.plot(t_smooth, d_smooth, 'r-', lw=2.5, 
                     label=f"Ajuste Torricelli\n$\\chi^2/ndof = {chi2_ndof:.2f}$")
            
            plt.title("Cinemática de Vaciado de Tanque (Ley de Torricelli)")
            plt.ylabel("Distancia Sensor-Agua $d_{real}$ (cm)")
            plt.xlabel("Tiempo $t$ (s)")
            plt.legend(loc='upper left')
            plt.grid(True, linestyle='--', alpha=0.6)
            plt.show()
            
        except Exception as e:
            print(f"Fallo en el ajuste de curva: {e}")
    else:
        print("El recorte dejó muy pocos datos para ajustar. Verifica los tiempos de la máscara.")
else:
    print("No se ejecutó captura de datos.")