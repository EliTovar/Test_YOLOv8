from ultralytics import YOLO

def train_yolo():
    # Cargar el modelo YOLOv8n preentrenado
    model = YOLO("yolov8n.pt")

    # Entrenar el modelo con tu dataset
    results = model.train(data="data.yaml", epochs=100, imgsz=640)

if __name__ == "__main__":
    train_yolo()
