import cv2
import numpy as np
from matplotlib import pyplot as plt

def detectar_bordes(ruta_imagen, umbral_min=100, umbral_max=200):
    """
    Carga una imagen, detecta sus bordes usando el algoritmo de Canny
    y devuelve la imagen binarizada con los bordes.

    Args:
        ruta_imagen (str): La ruta al archivo de la imagen.
        umbral_min (int): Umbral inferior para la detección de bordes de Canny.
        umbral_max (int): Umbral superior para la detección de bordes de Canny.

    Returns:
        tuple: Una tupla conteniendo (imagen_original, imagen_bordes) o (None, None) si hay error.
    """
    # Cargar la imagen del disco
    img = cv2.imread(ruta_imagen)

    # Verificar si la imagen se cargó correctamente
    if img is None:
        print(f"Error: No se pudo cargar la imagen en la ruta: {ruta_imagen}")
        return None, None

    # Convertir la imagen a escala de grises para el procesamiento
    img_gris = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # Aplicar el detector de bordes de Canny
    # Este algoritmo es bueno para encontrar contornos nítidos
    bordes = cv2.Canny(img_gris, umbral_min, umbral_max)

    return img, bordes

# --- Bloque principal para ejecutar el programa ---
if __name__ == "__main__":
    print("Iniciando el detector de bordes...")

    # --- ¡IMPORTANTE! ---
    # Cambia el nombre de archivo de abajo por el de tu imagen.
    # Asegúrate de que la imagen esté en la misma carpeta que este script.
    ruta_de_mi_imagen = 'Ace.jpg'

    # Llamar a la función para procesar la imagen
    imagen_original, imagen_bordes = detectar_bordes(ruta_de_mi_imagen)

    # Si la imagen se procesó correctamente, mostrar los resultados
    if imagen_original is not None and imagen_bordes is not None:
        print("Procesamiento exitoso. Mostrando resultados...")

        # Configurar la ventana de visualización con Matplotlib
        plt.figure(figsize=(10, 5))

        # Subplot 1: Imagen Original
        plt.subplot(1, 2, 1)
        # Se convierte de BGR a RGB para que Matplotlib muestre los colores correctos
        plt.imshow(cv2.cvtColor(imagen_original, cv2.COLOR_BGR2RGB))
        plt.title('Imagen Original')
        plt.xticks([]), plt.yticks([]) # Ocultar ejes

        # Subplot 2: Imagen de Bordes
        plt.subplot(1, 2, 2)
        plt.imshow(imagen_bordes, cmap='gray')
        plt.title('Bordes Detectados (Canny)')
        plt.xticks([]), plt.yticks([]) # Ocultar ejes

        # Mostrar la ventana
        plt.show()

        # Opcional: Guardar la imagen de bordes en un nuevo archivo
        # Extraer el nombre base para crear un nuevo nombre de archivo
        nombre_base = ruta_de_mi_imagen.split('.')[0]
        nombre_salida = f"bordes_{nombre_base}.png"

        cv2.imwrite(nombre_salida, imagen_bordes)
        print(f"Imagen de bordes guardada como: {nombre_salida}")