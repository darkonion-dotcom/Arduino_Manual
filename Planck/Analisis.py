import serial
import csv
import time

PUERTO = 'COM3'
BAUD_RATE = 9600
NUM_EXPERIMENTOS = 10
PUNTOS_POR_CURVA = 60 

# Menú para organizar los datos por color
print("¿Qué LED vas a medir en esta sesión?")
print("1. Rojo\n2. blanco\n3. Verde")
opcion = input("Elige una opción (1/2/3): ")
colores = {'1': 'rojo', '2': 'blanco', '3': 'verde'}
color = colores.get(opcion, 'rojo')

try:
    ser = serial.Serial(PUERTO, BAUD_RATE)
    time.sleep(2)
except Exception as e:
    print(f"Error en {PUERTO}: {e}")
    exit()

for exp in range(1, NUM_EXPERIMENTOS + 1):
    print(f"\n=========================================")
    print(f" EXPERIMENTO {exp}/10 - LED {color.upper()}")
    input(" 1. Pon el potenciómetro en 0V.\n 2. Presiona ENTER para calibrar el sensor...")
    
    datos = []
    puntos_tomados = 0
    voltaje_inicial = None
    ser.reset_input_buffer()
    
    # 1. Calibración inicial
    while voltaje_inicial is None:
        if ser.in_waiting > 0:
            try:
                linea = ser.readline().decode('utf-8').strip()
                v, i = map(float, linea.split(','))
                voltaje_inicial = v
                print(f"  -> Base calibrada en: {voltaje_inicial:.2f} V")
            except ValueError:
                pass
    
    print("  -> Esperando movimiento de la perilla...")
    
    # 2. Gatillo de arranque
    while True:
        if ser.in_waiting > 0:
            try:
                linea = ser.readline().decode('utf-8').strip()
                v, i = map(float, linea.split(','))
                
                if abs(v - voltaje_inicial) >= 0.05: 
                    print("  ¡Grabando datos!")
                    puntos_tomados += 1
                    datos.append([puntos_tomados, v, i])
                    time.sleep(0.2)
                    break 
            except ValueError:
                pass
    
    # 3. Captura
    while puntos_tomados < PUNTOS_POR_CURVA:
        if ser.in_waiting > 0:
            try:
                linea = ser.readline().decode('utf-8').strip()
                v, i = map(float, linea.split(','))
                puntos_tomados += 1
                datos.append([puntos_tomados, v, i])
                print(f"  Punto {puntos_tomados}/{PUNTOS_POR_CURVA}: {v:.2f} V | {i:.4f} mA")
                time.sleep(0.2)
            except ValueError:
                pass 
                
    nombre_archivo = f'datos_{color}_{exp}.csv'
    with open(nombre_archivo, mode='w', newline='') as file:
        writer = csv.writer(file)
        writer.writerow(['Punto', 'Voltaje (V)', 'Corriente (mA)'])
        writer.writerows(datos)

ser.close()
print(f"\n¡Los 10 experimentos del LED {color.upper()} han sido guardados!")