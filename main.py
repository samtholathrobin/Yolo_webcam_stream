from ultralytics import YOLO
import cv2
from utils import rgb_to_color,dom_rgb_mine
# model
model = YOLO("yolo-Weights/yolov8n.pt")
model_dress=YOLO("best.pt")
names=model.names
dnames=model_dress.names
p_id = list(names)[list(names.values()).index('person')]

# object classes
classNames = ["person", "bicycle", "car", "motorbike", "aeroplane", "bus", "train", "truck", "boat",
              "traffic light", "fire hydrant", "stop sign", "parking meter", "bench", "bird", "cat",
              "dog", "horse", "sheep", "cow", "elephant", "bear", "zebra", "giraffe", "backpack", "umbrella",
              "handbag", "tie", "suitcase", "frisbee", "skis", "snowboard", "sports ball", "kite", "baseball bat",
              "baseball glove", "skateboard", "surfboard", "tennis racket", "bottle", "wine glass", "cup",
              "fork", "knife", "spoon", "bowl", "banana", "apple", "sandwich", "orange", "broccoli",
              "carrot", "hot dog", "pizza", "donut", "cake", "chair", "sofa", "pottedplant", "bed",
              "diningtable", "toilet", "tvmonitor", "laptop", "mouse", "remote", "keyboard", "cell phone",
              "microwave", "oven", "toaster", "sink", "refrigerator", "book", "clock", "vase", "scissors",
              "teddy bear", "hair drier", "toothbrush"
              ]

num_person=''
color=''

# initialize the webcam
cap = cv2.VideoCapture('video (1080p).mp4') 
cap.set(3, 640)
cap.set(4, 480)

while True:
    success, img = cap.read()
    results = model(img, stream=True)
    dresses=model_dress(img,stream=True)
    for d in dresses:
        boxes = d.boxes
        for box in boxes:
            cls = int(box.cls[0])
            if (dnames[cls]=='shirt' or dnames[cls]=='Tshirt' or dnames[cls]=='jacket' or dnames[cls]=='dress'):
                x1, y1, x2, y2 = box.xyxy[0]
                x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2) 
                cropped_dress=img[y1:y2,x1:x2]
                r,g,b=dom_rgb_mine(cropped_dress)
                color=rgb_to_color(r,g,b)
                cv2.rectangle(img, (x1, y1), (x2, y2), (0,230,0), 3)
                org = [x1, y1]
                font = cv2.FONT_HERSHEY_SIMPLEX
                fontScale = 1
                colorout = (120,10,123)
                thickness = 2
                cv2.putText(img, dnames[cls]+" "+color, org, font, fontScale, colorout, thickness)
                
            
    for r in results:
        no_of_det=0
        for i in range(len(classNames)):
            if no_of_det<=15:
                if r.boxes.cls.tolist().count(i)!=0:
                    num="Number of "+classNames[i]+" : "+str(r.boxes.cls.tolist().count(i))
                    cv2.putText(img,num,[20,35+no_of_det*30],cv2.FONT_HERSHEY_SIMPLEX,0.75,(0, 0, 255),1)
                    no_of_det+=1
        boxes = r.boxes
        for box in boxes:
            # bounding box
            x1, y1, x2, y2 = box.xyxy[0]
            x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2) # convert to int values
            # put box in cam
            cv2.rectangle(img, (x1, y1), (x2, y2), (255,230,0), 3)
            #idx of class name in classNames
            cls = int(box.cls[0])
            org = [x1, y1]
            font = cv2.FONT_HERSHEY_SIMPLEX
            fontScale = 1
            color = (255,0,0)
            thickness = 2
            cv2.putText(img, classNames[cls], org, font, fontScale, color, thickness)

    cv2.imshow('Webcam', img)
    if cv2.waitKey(1) == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()