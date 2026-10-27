import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import linregress

# Parámetros físicos
e = 1.602e-19       
c = 299792458       
h_teorico = 6.62607015e-34 
RESISTENCIA = 330 # Ohmios

# CONFIGURACIÓN DE LEDs
leds = {
    'rojo':     {'color_graf': 'red',    'lambda': 654.5e-9},
    'blanco':   {'color_graf': 'silver', 'lambda': 450.0e-9},
    'verde':    {'color_graf': 'green',  'lambda': 525.0e-9}
}

NUM_EXPERIMENTOS = 10

# Ajuste de figura: Menos horizontal (ancho 7), más vertical (alto 8)
plt.figure(figsize=(7, 8))
h_globales = []

for color_nombre, config in leds.items():
    valores_v0 = []
    valores_h = []
    V_acumulados = []
    I_acumulados = []
    
    for exp in range(1, NUM_EXPERIMENTOS + 1):
        try:
            df = pd.read_csv(f'datos_{color_nombre}_{exp}.csv')
            V = df['Voltaje (V)'].values
            I = df['Corriente (mA)'].values
            
            V_acumulados.extend(V)
            I_acumulados.extend(I)
            
            plt.scatter(V, I, color=config['color_graf'], alpha=0.15, s=20)
            
            indices = np.where(I > 0.05)[0]
            if len(indices) < 2:
                continue
                
            V_aj = V[indices]
            I_aj = I[indices]
            
            pendiente, interseccion, _, _, _ = linregress(V_aj, I_aj)
            V0 = -interseccion / pendiente 
            valores_v0.append(V0)
            
            h_exp = (e * V0 * config['lambda']) / c
            valores_h.append(h_exp)
            
        except FileNotFoundError:
            pass 

    if len(valores_v0) > 0:
        v0_promedio = np.mean(valores_v0)
        h_promedio = np.mean(valores_h)
        h_globales.extend(valores_h)
        
        print(f"\n--- RESULTADOS LED {color_nombre.upper()} ---")
        print(f"U_0 Promedio: {v0_promedio:.3f} V")
        print(f"h calculada : {h_promedio:.8e} J.s")
        
        V_arr = np.array(V_acumulados)
        I_arr = np.array(I_acumulados)
        idx = np.where(I_arr > 0.05)[0]
        
        if len(idx) > 2:
            p, i, _, _, _ = linregress(V_arr[idx], I_arr[idx])
            V_line = np.linspace(v0_promedio, max(V_arr), 100)
            I_line = p * V_line + i
            
            plt.plot(V_line, I_line, color=config['color_graf'], linewidth=2, label=f'{color_nombre.capitalize()} ($U_0$={v0_promedio:.2f} V)')
            plt.plot([v0_promedio, v0_promedio], [0, p*v0_promedio + i], color=config['color_graf'], linestyle='--')

if len(h_globales) > 0:
    h_final_prom = np.mean(h_globales)
    h_final_std = np.std(h_globales)
    err = abs(h_final_prom - h_teorico) / h_teorico * 100
    
    print("\n=== CÁLCULO FINAL DE LA CONSTANTE DE PLANCK ===")
    print(f"\u0305h experimental: ({h_final_prom:.8e} \u00B1 {h_final_std:.8e}) J.s")
    print(f"Error relativo total: {err:.2f}%")
    
    # Cuadro de texto con la información final integrada en el gráfico
    texto_info = (f"Resistencia: {RESISTENCIA} $\\Omega$\n"
                  f"$\overline{{h}}$ exp: {h_final_prom:.3e} J$\cdot$s\n"
                  f"Error rel: {err:.2f}%")
    plt.text(1.52, max(I_acumulados)*0.85, texto_info, fontsize=10, 
             bbox=dict(facecolor='white', edgecolor='black', boxstyle='round,pad=0.5'))

# Estética final del gráfico
plt.axhline(0, color='black', linewidth=1.2) 
plt.title("Curvas Características I-V de LEDs (Rojo, Blanco, Verde)")
plt.xlabel("Voltaje (V)") # Etiqueta de eje X explícita con unidades
plt.ylabel("Corriente (mA)")
plt.ylim(bottom=-0.02) 

# Recorte horizontal para eliminar el espacio vacío inicial
plt.xlim(left=1.5) 

plt.legend(loc='upper left')

# Se omite plt.grid() para dejar el fondo limpio
plt.show()