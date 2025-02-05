import cv2
import numpy as np
from ultralytics import YOLO
from collections import OrderedDict

# -----------------------------
# Simple Centroid Tracker Class
# -----------------------------
class CentroidTracker:
    def __init__(self, maxDisappeared=50):
        self.nextObjectID = 0
        self.objects = OrderedDict()
        self.disappeared = OrderedDict()
        self.maxDisappeared = maxDisappeared

    def register(self, centroid, bbox):
        self.objects[self.nextObjectID] = (centroid, bbox)
        self.disappeared[self.nextObjectID] = 0
        self.nextObjectID += 1

    def deregister(self, objectID):
        del self.objects[objectID]
        del self.disappeared[objectID]

    def update(self, rects):
        # if no detections, mark existing objects as disappeared
        if len(rects) == 0:
            for objectID in list(self.disappeared.keys()):
                self.disappeared[objectID] += 1
                if self.disappeared[objectID] > self.maxDisappeared:
                    self.deregister(objectID)
            return self.objects

        # Compute centroids for input rectangles
        inputCentroids = np.zeros((len(rects), 2), dtype="int")
        for i, (startX, startY, endX, endY) in enumerate(rects):
            cX = int((startX + endX) / 2.0)
            cY = int((startY + endY) / 2.0)
            inputCentroids[i] = (cX, cY)

        # if no objects are being tracked, register all centroids
        if len(self.objects) == 0:
            for i in range(0, len(inputCentroids)):
                self.register(inputCentroids[i], rects[i])
        else:
            # grab existing object IDs and centroids
            objectIDs = list(self.objects.keys())
            objectCentroids = [self.objects[objectID][0] for objectID in objectIDs]

            # compute distance between each pair of object centroids and new centroids
            D = np.linalg.norm(np.array(objectCentroids)[:, np.newaxis] - inputCentroids, axis=2)
            rows = D.min(axis=1).argsort()
            cols = D.argmin(axis=1)[rows]

            usedRows = set()
            usedCols = set()

            # assign the new centroid to an existing object based on minimum distance
            for (row, col) in zip(rows, cols):
                if row in usedRows or col in usedCols:
                    continue
                objectID = objectIDs[row]
                self.objects[objectID] = (inputCentroids[col], rects[col])
                self.disappeared[objectID] = 0
                usedRows.add(row)
                usedCols.add(col)

            # mark unmatched objects as disappeared
            unusedRows = set(range(0, D.shape[0])).difference(usedRows)
            unusedCols = set(range(0, D.shape[1])).difference(usedCols)

            if D.shape[0] >= D.shape[1]:
                for row in unusedRows:
                    objectID = objectIDs[row]
                    self.disappeared[objectID] += 1
                    if self.disappeared[objectID] > self.maxDisappeared:
                        self.deregister(objectID)
            else:
                for col in unusedCols:
                    self.register(inputCentroids[col], rects[col])
        return self.objects

# -----------------------------
# Load YOLO Model from a .pt file
# -----------------------------
# Make sure that the path './misc/yolov5su.pt' is correct.
model = YOLO('./misc/yolov5su.pt')  # Specify the correct model file path

# If your model includes class names, they might be available as model.names.
# Otherwise, you can define a default list.
if not hasattr(model, 'names'):
    model.names = {i: f'class_{i}' for i in range(1000)}

# -----------------------------
# Initialize Video Stream and Tracker
# -----------------------------
cap = cv2.VideoCapture(0)
tracker = CentroidTracker(maxDisappeared=40)

while True:
    ret, frame = cap.read()
    if not ret:
        break

    # Perform inference on the current frame.
    # The model returns a list of results. We use the first result.
    results = model(frame)
    
    rects = []
    # Access predictions; for YOLOv8 API, predictions are stored in .boxes
    if results and results[0].boxes is not None:
        boxes = results[0].boxes
        for box in boxes:
            # Retrieve bounding box coordinates in xyxy format and cast to int.
            # box.xyxy is a tensor of shape (1, 4).
            xyxy = box.xyxy[0].cpu().numpy().astype(int)
            x1, y1, x2, y2 = xyxy
            rects.append((x1, y1, x2, y2))
            # Retrieve class and confidence if available.
            cls = int(box.cls[0].cpu().numpy())
            label = model.names.get(cls, "object")
            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
            cv2.putText(frame, label, (x1, y1 - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

    # Update tracker with the current set of bounding boxes.
    objects = tracker.update(rects)
    for objectID, (centroid, bbox) in objects.items():
        text = f"ID {objectID}"
        cv2.putText(frame, text, (centroid[0] - 10, centroid[1] - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)
        cv2.circle(frame, (centroid[0], centroid[1]), 4, (0, 0, 255), -1)

    cv2.imshow("YOLO Object Detection & Tracking", frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
