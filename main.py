import cv2
import numpy as np
import pandas as pd
import pytesseract
from ultralytics import YOLO

pytesseract.pytesseract.tesseract_cmd = r'C:\Users\william_tan\AppData\Local\Programs\Tesseract-OCR\tesseract.exe'  # Update this path if your Tesseract is installed elsewhere

def detect_vehicles(frame, model):
    # Use YOLOv8 via ultralytics to detect vehicles and classify type
    results = model(frame)
    vehicle_classes = {
        2: 'car',
        3: 'motorcycle',
        5: 'bus',
        7: 'truck'
    }  # COCO: car, motorcycle, bus, truck
    boxes = []
    types = []
    for r in results:
        for box in r.boxes:
            cls = int(box.cls)
            if cls in vehicle_classes:
                x1, y1, x2, y2 = map(int, box.xyxy[0])
                boxes.append((x1, y1, x2 - x1, y2 - y1))
                types.append(vehicle_classes[cls])
    return list(zip(boxes, types))

def recognize_plate(plate_img):
    # Use Tesseract OCR to recognize text from plate image
    gray = cv2.cvtColor(plate_img, cv2.COLOR_BGR2GRAY)
    # Optional: thresholding for better OCR
    _, thresh = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    config = '--psm 7'  # Assume a single line of text
    text = pytesseract.image_to_string(thresh, config=config)
    return text.strip()

def get_vehicle_color(vehicle_img):
    # Simple color detection (dominant color)
    avg_color = np.mean(vehicle_img, axis=(0, 1))
    b, g, r = avg_color
    if r > g and r > b:
        return "Red"
    elif g > r and g > b:
        return "Green"
    elif b > r and b > g:
        return "Blue"
    else:
        return "Other"

def main(video_path, output_excel):
    cap = cv2.VideoCapture(video_path)
    model = YOLO('yolov8n.pt')
    results = []
    frame_num = 0
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
        vehicles = detect_vehicles(frame, model)
        # Draw bounding boxes and labels for each detected vehicle
        for ((x, y, w, h), vehicle_type) in vehicles:
            # Draw bounding box with color based on vehicle type
            if vehicle_type == 'car':
                color = (0, 255, 0)      # Green
            elif vehicle_type == 'motorcycle':
                color = (255, 0, 0)      # Blue
            elif vehicle_type == 'bus':
                color = (0, 255, 255)    # Yellow
            elif vehicle_type == 'truck':
                color = (0, 0, 255)      # Red
            else:
                color = (255, 255, 255)  # White (fallback)
            cv2.rectangle(frame, (x, y), (x + w, y + h), color, 2)
            # Draw label
            label = f"{vehicle_type}"
            cv2.putText(frame, label, (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)
            vehicle_img = frame[y:y+h, x:x+w]
            # Plate localization: fallback to lower part of vehicle as plate_img
            plate_img = vehicle_img[int(vehicle_img.shape[0]*0.6):, int(vehicle_img.shape[1]*0.2):int(vehicle_img.shape[1]*0.8)]
            # Only run OCR if plate_img is not empty
            if plate_img.size > 0:
                plate_text = recognize_plate(plate_img)
            else:
                plate_text = ""
            vehicle_color = get_vehicle_color(vehicle_img)
            results.append({
                'Frame': frame_num,
                'Number Plate': plate_text,
                'Vehicle Type': vehicle_type,
                'Vehicle Color': vehicle_color
            })
        # Display the number plate number at the top right of the screen
        if results and results[-1]['Number Plate']:
            np_text = results[-1]['Number Plate']
            (tw, th), _ = cv2.getTextSize(np_text, cv2.FONT_HERSHEY_SIMPLEX, 1.2, 3)
            cv2.putText(frame, np_text, (frame.shape[1] - tw - 20, 40), cv2.FONT_HERSHEY_SIMPLEX, 1.2, (255,255,255), 3)
        # Display the frame with bounding boxes and labels
        cv2.imshow('Vehicle Detection', frame)
        key = cv2.waitKey(1) & 0xFF
        if key == ord('q') or key == 27:  # 27 is ESC
            break
        frame_num += 1
    cap.release()
    cv2.destroyAllWindows()
    df = pd.DataFrame(results)
    df.to_excel(output_excel, index=False)

if __name__ == "__main__":
    main("kaggle.mp4", "output.xlsx")
