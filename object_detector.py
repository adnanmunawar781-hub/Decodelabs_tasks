# =====================================================================
# PROJECT 4: BUILDING THE MACHINE'S OPTIC NERVE (OBJECT DETECTION)
# Execution Path 2: OpenCV & MobileNet-SSD
# =====================================================================

import cv2
import numpy as np

print("👁️ Initializing Machine's Optic Nerve...")

# --- SETUP: Load the Pre-trained Model (Transfer Learning) ---
# Classes that MobileNet-SSD is trained to detect
CLASSES = ["background", "aeroplane", "bicycle", "bird", "boat",
           "bottle", "bus", "car", "cat", "chair", "cow", "diningtable",
           "dog", "horse", "motorbike", "person", "pottedplant", "sheep",
           "sofa", "train", "tvmonitor"]

# Load the network using standard OpenCV DNN module
try:
    print("[INFO] Loading model architecture and weights...")
    net = cv2.dnn.readNetFromCaffe("MobileNetSSD.prototxt", "MobileNetSSD.caffemodel")
except Exception as e:
    print("❌ ERROR: Missing Model Files! Make sure .prototxt and .caffemodel are in the folder.")
    exit()

# --- INPUT: Load the Visual Matrix ---
image_path = "test_image.jpg" # Change this to your image name
image = cv2.imread(image_path)
if image is None:
    print(f"❌ ERROR: Could not find '{image_path}'. Add an image to test.")
    exit()

# Extract height (H) and width (W) to scale the bounding boxes later
(H, W) = image.shape[:2]

# ==========================================
# GATEKEEPER 2: PRE-PROCESSING INTEGRITY (Blob Construction)
# ==========================================
# We don't pass raw images to the model. We create a 4D Blob.
# Scaling it to 300x300 and performing mean subtraction (127.5) to normalize lighting.
blob = cv2.dnn.blobFromImage(cv2.resize(image, (300, 300)), 0.007843, (300, 300), 127.5)

# ==========================================
# PROCESS: Passing Blob through the Network
# ==========================================
print("[INFO] Executing Forward Pass through MobileNet-SSD...")
net.setInput(blob)
detections = net.forward() # The machine makes its predictions!

# ==========================================
# OUTPUT: Decoding the Matrix & Confidence Filtering
# ==========================================
# Loop over the detections
for i in np.arange(0, detections.shape[2]):
    
    # Extract the probability (confidence) associated with the prediction
    confidence = detections[0, 0, i, 2]
    
    # GATEKEEPER 3 & 4: THE 80% CONFIDENCE FILTER & VISUAL CONFIRMATION
    if confidence >= 0.80:
        # Extract the index of the class label
        idx = int(detections[0, 0, i, 1])
        label = CLASSES[idx]
        
        # Calculate the Bounding Box coordinates (Normalized -> Actual Pixels)
        # Bounding box array: [startX, startY, endX, endY]
        box = detections[0, 0, i, 3:7] * np.array([W, H, W, H])
        (startX, startY, endX, endY) = box.astype("int")
        
        # Draw the Bounding Box (Red color) and the Label (Blue color)
        cv2.rectangle(image, (startX, startY), (endX, endY), (0, 0, 255), 2)
        
        # Format the label string with confidence percentage
        text = f"{label}: {confidence * 100:.2f}%"
        print(f"✅ DETECTED: {text}")
        
        # Put text above the bounding box
        y = startY - 15 if startY - 15 > 15 else startY + 15
        cv2.putText(image, text, (startX, y), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 0, 0), 2)

# --- DISPLAY THE RESULT ---

print("[INFO] Rendering the visual output...")


cv2.namedWindow("Machine Perception (Project 4)", cv2.WINDOW_NORMAL)
cv2.resizeWindow("Machine Perception (Project 4)", 800, 600)

cv2.imshow("Machine Perception (Project 4)", image)
cv2.waitKey(0) # Wait until the user presses any key
cv2.destroyAllWindows()