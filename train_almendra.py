"""
train_yolo.py
Entrena un modelo YOLOv8 en tu dataset de frutas (incluido kiwi).
Requisitos: pip install ultralytics
Ejecutar: python train_yolo.py
"""

from ultralytics import YOLO

def main():
    # Cargar un modelo preentrenado (puedes usar yolov8n.pt, yolov8s.pt, yolov8m.pt, etc.)
    # "n" es más rápido pero menos preciso, "m" o "l" son más precisos
    model = YOLO("yolov8n.pt")

    # Entrenar el modelo con tu dataset
    results = model.train(
        data="data.yaml",        # ruta a tu archivo de configuración
        epochs=20,              # número de épocas (puedes ajustar)
        imgsz=640,               # tamaño de imagen (ajustable, 640 es estándar)
        batch=16,                # tamaño de lote (ajústalo según tu GPU)
        name="almendra_end",   # nombre del experimento / carpeta de resultados
        device="cpu",                # 0 = usar GPU, 'cpu' si no tienes
        project="runs/train",    # carpeta donde se guardarán resultados
        exist_ok=True            # sobrescribir si ya existe
    )

    # Evaluar el modelo
    model.val()

    # Exportar modelo entrenado a .pt
    model.export(format="pt")

if __name__ == "__main__":
    main()
