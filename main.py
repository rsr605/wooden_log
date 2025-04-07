from ultralytics import YOLO

model = YOLO('last.pt')
results = model.predict(source='a.jpg',save=True)
