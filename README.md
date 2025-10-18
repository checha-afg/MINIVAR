# ⚽ Detector de Fuera de Juego Inteligente

Proyecto desarrollado por **César Flores** como parte del curso **Seminario de Tecnologías de Información**, Universidad Mariano Gálvez de Guatemala (Sede Jocotenango).

Este sistema utiliza **Inteligencia Artificial** y el modelo **YOLOv8** para analizar videos de partidos de fútbol y detectar posibles jugadas de **fuera de juego**.

---

## Características del proyecto
- Entrenamiento con **YOLOv8 (Ultralytics)**  
- Procesamiento de video con **OpenCV**
- Identificación de **jugadores, porteros, árbitros y pelota**
- Detección automática de líneas de fuera de juego
- Pausa automática del video al detectar una situación sospechosa

---

## Requisitos previos

Asegúrate de tener instalado:
- **Python 3.10 o superior**
- **pip** actualizado
- (Recomendado) **NVIDIA GPU con CUDA** o cualquier otra tarjeta grafica con más de 4GB de VRAM

---

## Instalación de dependencias

Ejecuta los siguientes comandos en tu terminal o PowerShell dentro del directorio del proyecto:

## ⬇Instalar dependencias principales
pip install ultralytics

pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118

pip install opencv-python numpy
