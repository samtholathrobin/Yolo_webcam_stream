import unittest
from ultralytics import YOLO

model = YOLO("yolo-Weights/yolov8n.pt")

def yolo_obj_detect(img):
    results = model(img, stream=True)
    l=[]
    for i in results:
        boxes=i.boxes
        for box in boxes:
            x1, y1, x2, y2 = box.xyxy[0]
            x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)
            l.append([x1, y1, x2, y2])
    return l

class MyTestCase(unittest.TestCase):
   
    def test_boxes1(self):
       result = yolo_obj_detect('people_in_street_1.jpg')  
       self.assertEqual(result, [[38, 152, 101, 358], [432, 177, 486, 349], [287, 188, 375, 422], [545, 183, 611, 349], [403, 168, 439, 283], [137, 176, 195, 327], [500, 167, 561, 308], [239, 172, 294, 318], [0, 166, 34, 296], [603, 170, 644, 296], [195, 157, 238, 282], [370, 167, 393, 236], [36, 255, 60, 294], [386, 168, 409, 250], [244, 194, 273, 255], [244, 195, 275, 254], [182, 235, 197, 265], [105, 169, 143, 281], [475, 211, 513, 277], [229, 161, 263, 266]])
       
    def test_det1(self):
       result=yolo_obj_detect('people_in_street_1.jpg')
       r=len(result)
       self.assertEqual(r,20)

    def test_boxes2(self):
       result = yolo_obj_detect('people_in_street_2.jpg')  
       self.assertEqual(result, [[1, 50, 42, 159], [200, 33, 249, 180], [137, 51, 190, 172], [76, 40, 117, 170], [51, 57, 89, 162], [184, 72, 202, 145], [185, 72, 202, 122], [169, 84, 191, 118]])
       
    def test_det2(self):
       result=yolo_obj_detect('people_in_street_2.jpg')
       r=len(result)
       self.assertEqual(r,8)

    def test_boxes3(self):
       result = yolo_obj_detect('people_in_street_3.jpg')  
       self.assertEqual(result, [[136, 0, 209, 172], [244, 21, 286, 164], [225, 35, 247, 104], [67, 23, 99, 113], [115, 37, 136, 104], [149, 103, 186, 142], [146, 66, 204, 142]])

    def test_det3(self):
       result=yolo_obj_detect('people_in_street_3.jpg')
       r=len(result)
       self.assertEqual(r,7)

