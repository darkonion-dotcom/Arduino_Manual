import serial
import csv
import time

PUERTO = 'COM3'
BAUD_RATE = 9600
NUM_EXPERIMENTOS = 10
PUNTOS_POR_CURVA = 25 # Cantidad de lecturas para formar bien la curva en cada intento

try:
    ser = serial.Serial(PUERTO, BAUD_RATE)
    time.sleep(2)
except Exception as e:
    print(f"Error en {PUERTO}: {e}")
    exit()

for exp in range(1, NUM_EXPERIMENTOS + 1):
    print(f"\n=========================================")
    print(f" PREPARANDO EXPERIMENTO {exp} DE {NUM_EXPERIMENTOS}")
    input(" 1. Gira el potenciómetro a 0V (totalmente apagado).\n 2. Presiona ENTER y empieza a girar LENTAMENTE hacia 5V...")
    
    datos = []
    puntos_tomados = 0
    
    while puntos_tomados < PUNTOS_POR_CURVA:
        if ser.in_waiting > 0:
            try:
                linea = ser.readline().decode('utf-8').strip()
                v, i = map(float, linea.split(','))
                
                puntos_tomados += 1
                datos.append([puntos_tomados, v, i])
                print(f"  Punto {puntos_tomados}/{PUNTOS_POR_CURVA}: {v:.2f} V | {i:.2f} mA")
                
                time.sleep(0.2) # Lee cada 0.2 segundos para capturar bien la subida
            except ValueError:
                pass 
                
    nombre_archivo = f'experimento_{exp}.csv'
    with open(nombre_archivo, mode='w', newline='') as file:
        writer = csv.writer(file)
        writer.writerow(['Punto', 'Voltaje (V)', 'Corriente (mA)'])
        writer.writerows(datos)
    
    print(f"--> Curva guardada en {nombre_archivo}")

ser.close()
print("\n¡Los 10 experimentos han sido completados y guardados!")