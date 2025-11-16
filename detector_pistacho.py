from ultralytics import YOLO
import cv2

# Cargar el modelo entrenado
model = YOLO(r"C:\Users\LENOVO\Desktop\imagenes_entrenamiento\pistachio_end\runs\train\almendra_end\weights\best.pt")

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