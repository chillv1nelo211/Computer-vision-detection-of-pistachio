from ultralytics import YOLO
import cv2

# Obtener la ruta del directorio actual del script
script_dir = os.path.dirname(os.path.abspath(__file__))
# Construir la ruta relativa al modelo
model_path = os.path.join(script_dir, "runs/train/almendra_end/weights/best.pt")

# Cargar el modelo entrenado
model = YOLO(model_path)

# Inicializar la cámara (0 = cámara por defecto)
cap = cv2.VideoCapture(0)

if not cap.isOpened():
    print("No se pudo acceder a la cámara.")
    exit()

print("Cámara iniciada. Presiona 'q' para salir.")

while True:
    ret, frame = cap.read()
    if not ret:
        print("Error al capturar el video")
        break

    # Ejecutar la detección en el frame
    results = model(frame)

    # Mostrar los resultados con los cuadros dibujados
    annotated_frame = results[0].plot()

    cv2.imshow("Detección en tiempo real", annotated_frame)

    # Salir al presionar 'q'
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Liberar la cámara y cerrar ventanas
cap.release()
cv2.destroyAllWindows()