from ultralytics import YOLO

def main():
    # Cargar modelo base preentrenado
    model = YOLO("yolov8m.pt")  # Modelo mediano, buena precisión sin ser gigante

    # Entrenamiento optimizado
    model.train(
        data="dataset/data.yaml",  # Tu dataset
        epochs=60,                 # Número suficiente de epochs
        imgsz=800,                 # Resolución alta, mejora detección
        batch=8,                   # GPU 8GB, batch razonable
        device="cuda",             # GPU
        rect=True,                 # Anchura/altura variable optimiza dataloader
        workers=4,                 # Paralelismo de carga de datos
        name="train_gpu_v11",      # Carpeta de resultados
        project="runs/detect",

        # Optimización del entrenamiento
        lr0=0.001,
        augment=True,              # Habilita augmentaciones automáticas

        # Aumentos de datos específicos
        mosaic=0.3,
        mixup=0.1,
        copy_paste=0.2,
        hsv_h=0.015,
        hsv_s=0.7,
        hsv_v=0.4,
        degrees=5.0,
        translate=0.1,
        scale=0.5,
        shear=0.0,
        perspective=0.0,
        flipud=0.0,
        fliplr=0.5,

        # Precisión y estabilidad
        freeze=[0, 1, 2],          # Congelar primeras capas del backbone
        amp=True,                  # Mixed Precision Automatic (ahorra VRAM)
    )

if __name__ == "__main__":
    main()
