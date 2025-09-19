import tkinter as tk
from tkinter import filedialog, messagebox, ttk
import cv2
import numpy as np
from matplotlib import pyplot as plt

# -----------------------------------------------------------------------------
# Lógica de Procesamiento de Imágenes (Backend)
# -----------------------------------------------------------------------------
def detectar_bordes(ruta_imagen, umbral_min=100, umbral_max=200):
    """Carga una imagen y detecta sus bordes con el algoritmo de Canny."""
    img_color = cv2.imread(ruta_imagen)
    if img_color is None:
        return None, None
    img_gris = cv2.cvtColor(img_color, cv2.COLOR_BGR2GRAY)
    bordes = cv2.Canny(img_gris, umbral_min, umbral_max)
    return img_color, bordes

# --- Biblioteca de Mapeos ---
def mapeo_lineal(puntos_z, a=1.5, b=0): return a * puntos_z + b
def mapeo_cuadratico(puntos_z): return puntos_z ** 2
def mapeo_inverso(puntos_z): return 1 / (puntos_z + 1e-8)
def mapeo_exponencial(puntos_z): return np.exp(puntos_z) # Ya no se divide por 100
def mapeo_bilineal(puntos_z, a=1, b=0, c=0, d=1):
    denominador = c * puntos_z + d
    denominador[np.abs(denominador) < 1e-8] = 1e-8
    return (a * puntos_z + b) / denominador

mapeos_disponibles = {
    "Lineal": mapeo_lineal,
    "Cuadrático": mapeo_cuadratico,
    "Inverso": mapeo_inverso,
    "Exponencial": mapeo_exponencial,
    "Bilineal": mapeo_bilineal,
}

# -----------------------------------------------------------------------------
# Clase de la Aplicación con Tkinter (Frontend)
# -----------------------------------------------------------------------------
class MappingApp:
    def __init__(self, root):
        self.root = root
        self.root.title("Art-Attack: Mapeos Complejos")
        self.root.geometry("450x320")

        self.filepath = None
        self.mapeo_seleccionado = tk.StringVar(value="Lineal")

        # --- Frame para selección de archivo ---
        file_frame = ttk.LabelFrame(root, text="1. Seleccionar Imagen")
        file_frame.pack(padx=10, pady=10, fill="x")

        self.select_button = ttk.Button(file_frame, text="Abrir Imagen", command=self.abrir_archivo)
        self.select_button.pack(side="left", padx=5, pady=5)
        self.path_label = ttk.Label(file_frame, text="Ningún archivo seleccionado")
        self.path_label.pack(side="left", padx=5)

        # --- Frame para selección de mapeo ---
        map_frame = ttk.LabelFrame(root, text="2. Elegir Mapeo")
        map_frame.pack(padx=10, pady=5, fill="x", expand=True)

        for nombre_mapeo in mapeos_disponibles.keys():
            rb = ttk.Radiobutton(map_frame, text=nombre_mapeo, variable=self.mapeo_seleccionado, value=nombre_mapeo)
            rb.pack(anchor="w", padx=10)
        
        self.cascade_rb = ttk.Radiobutton(map_frame, text="Cascada (ej: Cuadrático,Inverso)", variable=self.mapeo_seleccionado, value="Cascada")
        self.cascade_rb.pack(anchor="w", padx=10)
        self.cascade_entry = ttk.Entry(map_frame)
        self.cascade_entry.pack(fill="x", padx=25, pady=2)

        # --- Botón para aplicar el mapeo ---
        self.apply_button = ttk.Button(root, text="Aplicar Mapeo y Visualizar", command=self.aplicar_mapeo)
        self.apply_button.pack(pady=10)

    def abrir_archivo(self):
        self.filepath = filedialog.askopenfilename(
            title="Selecciona una imagen",
            filetypes=(("Archivos de Imagen", "*.jpg *.jpeg *.png"), ("Todos los archivos", "*.*"))
        )
        if self.filepath:
            self.path_label.config(text=self.filepath.split('/')[-1])

    def aplicar_mapeo(self):
        if not self.filepath:
            messagebox.showerror("Error", "Por favor, selecciona una imagen primero.")
            return

        _, imagen_bordes = detectar_bordes(self.filepath)
        if imagen_bordes is None:
            messagebox.showerror("Error", "No se pudo cargar o procesar la imagen.")
            return

        filas, columnas = np.where(imagen_bordes > 0)
        puntos_z = (columnas - imagen_bordes.shape[1] / 2) + 1j * (imagen_bordes.shape[0] / 2 - filas)
        
        # --- NORMALIZACIÓN DE COORDENADAS ---
        # Se reescala la figura para que siempre esté en un rango de -1 a 1.
        max_distancia = np.max(np.abs(puntos_z))
        if max_distancia > 0:
            puntos_z_normalizados = puntos_z / max_distancia
        else:
            puntos_z_normalizados = puntos_z
        # --- FIN DE LA NORMALIZACIÓN ---
        
        puntos_transformados = None
        titulo = ""
        seleccion = self.mapeo_seleccionado.get()

        try:
            if seleccion in mapeos_disponibles:
                puntos_a_mapear = puntos_z_normalizados
                # Parámetros de ejemplo para la nueva escala normalizada
                if seleccion == "Bilineal":
                    puntos_transformados = mapeos_disponibles[seleccion](puntos_a_mapear, a=1, c=0.8, d=1)
                else:
                    puntos_transformados = mapeos_disponibles[seleccion](puntos_a_mapear)
                titulo = f"Mapeo {seleccion}"
            
            elif seleccion == "Cascada":
                secuencia_str = self.cascade_entry.get()
                if not secuencia_str:
                    messagebox.showerror("Error", "La secuencia de cascada no puede estar vacía.")
                    return
                
                nombres_secuencia = [s.strip() for s in secuencia_str.split(',')]
                puntos_actuales = puntos_z_normalizados.copy()
                
                for nombre in nombres_secuencia:
                    if nombre in mapeos_disponibles:
                        puntos_actuales = mapeos_disponibles[nombre](puntos_actuales)
                    else:
                        raise ValueError(f"Mapeo desconocido en la secuencia: '{nombre}'")
                
                puntos_transformados = puntos_actuales
                titulo = "Cascada: " + " -> ".join(nombres_secuencia)

            if puntos_transformados is not None:
                self.mostrar_resultado(imagen_bordes, puntos_transformados, titulo)

        except ValueError as e:
            messagebox.showerror("Error de Secuencia", str(e))
        except Exception as e:
            messagebox.showerror("Error Inesperado", f"Ocurrió un error durante el mapeo: {e}")

    def mostrar_resultado(self, bordes_originales, puntos_w, titulo):
        plt.figure(figsize=(12, 6))
        
        plt.subplot(1, 2, 1)
        plt.imshow(bordes_originales, cmap='gray')
        plt.title('Bordes Originales')
        plt.axis('off')

        plt.subplot(1, 2, 2)
        plt.scatter(puntos_w.real, puntos_w.imag, s=1, color='blue')
        plt.title(titulo)
        plt.xlabel('Eje Real (u)')
        plt.ylabel('Eje Imaginario (v)')
        plt.axis('equal')
        
        plt.tight_layout()
        plt.show()

# --- Punto de Entrada del Script ---
if __name__ == "__main__":
    root = tk.Tk()
    app = MappingApp(root)
    root.mainloop()