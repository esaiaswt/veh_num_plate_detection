# Vehicle Number Plate Detection

This project uses YOLOv8 and Google Tesseract OCR to detect vehicles, recognize their number plates, classify vehicle type and color, and save the results to an Excel file. It also displays the video with bounding boxes and labels for each detected vehicle and number plate.

## Features
- Detects vehicles (car, motorcycle, bus, truck) in video frames using YOLOv8
- Recognizes number plates using Tesseract OCR
- Classifies vehicle color (red, green, blue, other)
- Displays bounding boxes and labels in real time
- Saves results (frame, number plate, vehicle type, color) to Excel

## Requirements
- Python 3.8+
- [Tesseract OCR](https://github.com/tesseract-ocr/tesseract) (install and set the path in `main.py`)
- See `requirements.txt` for Python dependencies
- YOLOv8 model file (`yolov8n.pt`) in the project directory

## Usage
1. Install dependencies:
   ```sh
   pip install -r requirements.txt
   ```
2. Install Tesseract OCR and update the path in `main.py` if needed:
   ```python
   pytesseract.pytesseract.tesseract_cmd = r'C:\Path\To\tesseract.exe'
   ```
3. Place your video file (e.g., `kaggle.mp4`) and YOLOv8 model file (`yolov8n.pt`) in the project directory.
4. Run the script:
   ```sh
   python main.py
   ```
5. Press `q` or `ESC` to quit the video display.

## Output
- Annotated video display with bounding boxes and labels
- Results saved to `output.xlsx`

## Notes
- The number plate region is estimated as the lower part of the vehicle bounding box.
- For best OCR results, ensure Tesseract is properly installed and the path is set.
- The script will display the recognized number plate at the top right of the screen for each frame.

## Example
```
Frame | Number Plate | Vehicle Type | Vehicle Color
--------------------------------------------------
   10 | WXY1234      | car         | Red
   11 | BCD5678      | truck       | Blue
```

---

**Author:** [Your Name]
