import cv2
import numpy as np
import pandas as pd
import pytesseract
from ultralytics import YOLO
import imutils
from skimage import measure

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

def find_plate_region(plate_img):
    # Step 1: Blur and grayscale
    img_blur = cv2.GaussianBlur(plate_img, (7, 7), 0)
    gray = cv2.cvtColor(img_blur, cv2.COLOR_BGR2GRAY)
    # Step 2: Sobel vertical edge detection
    sobelx = cv2.Sobel(gray, cv2.CV_8U, 1, 0, ksize=3)
    # Step 3: Otsu's thresholding
    _, threshold_img = cv2.threshold(sobelx, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    # Step 4: Morphological closing
    element = cv2.getStructuringElement(cv2.MORPH_RECT, (22, 3))
    morph_img = cv2.morphologyEx(threshold_img, cv2.MORPH_CLOSE, element)
    # Step 5: Find contours
    contours, _ = cv2.findContours(morph_img, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    plate_candidate = None
    max_area = 0
    for cnt in contours:
        x, y, w, h = cv2.boundingRect(cnt)
        area = w * h
        ratio = w / float(h)
        if 4500 < area < 30000 and 2.5 < ratio < 6:
            if area > max_area:
                max_area = area
                plate_candidate = (x, y, w, h)
    if plate_candidate:
        x, y, w, h = plate_candidate
        return plate_img[y:y+h, x:x+w]
    else:
        return plate_img  # fallback

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
            # Only process cars for number plate detection/recognition
            if vehicle_type != 'car':
                continue
            # Draw bounding box with color based on vehicle type
            color = (0, 255, 0)      # Green
            cv2.rectangle(frame, (x, y), (x + w, y + h), color, 2)
            # Draw label
            label = f"{vehicle_type}"
            cv2.putText(frame, label, (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)
            vehicle_img = frame[y:y+h, x:x+w]
            # Use bottom half of vehicle_img for plate detection
            vh, vw = vehicle_img.shape[:2]
            bottom_half = vehicle_img[vh//2:]
            # Use GeeksforGeeks method to find plate region
            plate_region = find_plate_region(bottom_half)
            # Only run OCR if a plate region was detected (not fallback)
            if plate_region.shape[0] != bottom_half.shape[0] or plate_region.shape[1] != bottom_half.shape[1]:
                cv2.imshow('Plate Region Before OCR', plate_region)
                plate_text = recognize_plate(plate_region)
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
