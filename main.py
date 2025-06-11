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
            # Only process trucks for number plate detection/recognition
            if vehicle_type != 'car':
                continue
            # Draw bounding box with color based on vehicle type
            color = (0, 0, 255)      # Red for truck
            cv2.rectangle(frame, (x, y), (x + w, y + h), color, 2)
            # Draw label
            label = f"{vehicle_type}"
            cv2.putText(frame, label, (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)
            vehicle_img = frame[y:y+h, x:x+w]
            # Use the full detection box of the truck for plate detection
            plate_region = find_plate_region(vehicle_img)
            # Only run OCR if a plate region was detected (not fallback)
            if plate_region.shape[0] != vehicle_img.shape[0] or plate_region.shape[1] != vehicle_img.shape[1]:
                # Magnify the detected plate region 3x before OCR
                plate_region_magnified = cv2.resize(plate_region, None, fx=3, fy=3, interpolation=cv2.INTER_CUBIC)
                # Only show the process image window when a truck is detected
                cv2.imshow('Plate Region Before OCR', plate_region_magnified)
                plate_text = recognize_plate(plate_region_magnified)
                # Save the detected plate region as PNG only if OCR output is alphanumeric
                if any(c.isalnum() for c in plate_text):
                    save_path = f"detected_plate_{frame_num}.png"
                    cv2.imwrite(save_path, plate_region_magnified)
                # Fast forward: skip to next frame quickly when truck is detected
                key = cv2.waitKey(10) & 0xFF  # 10ms wait instead of 1ms
            else:
                # Hide the process image window if no truck/plate is detected
                try:
                    cv2.destroyWindow('Plate Region Before OCR')
                except cv2.error:
                    pass
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
        # If a plate region is detected, also show it on the main display
        if 'plate_region_magnified' in locals() and plate_region_magnified is not None:
            # Resize the plate region to fit a small box (e.g., 200x60)
            preview = cv2.resize(plate_region_magnified, (200, 60), interpolation=cv2.INTER_AREA)
            # Place the preview at the top left corner of the main frame
            frame[10:70, 10:210] = preview
        # Display the frame with bounding boxes and labels
        # Resize frame to fit window (e.g., width=900, keep aspect ratio)
        display_width = 900
        h, w = frame.shape[:2]
        scale = display_width / w
        display_frame = cv2.resize(frame, (display_width, int(h * scale)), interpolation=cv2.INTER_AREA)
        cv2.imshow('Vehicle Detection', display_frame)
        # Only wait for key if not already handled in fast forward
        if not ('key' in locals() and key is not None):
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
