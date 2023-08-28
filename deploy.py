import os
import sys
from pathlib import Path
import time
import numpy as np
import cv2
import torch
import torch.backends.cudnn as cudnn
import io
import base64
import datetime
import tkinter as tk
from tkinter import Tk
from tkinter import filedialog
import cv2
from PIL import Image, ImageTk
import torch
import numpy as np
import os

root = Tk()
root.bind("<Escape>", lambda e: root.quit())
root.geometry("800x800")

# Rest of the code
import os
import sys
from pathlib import Path
import time
import numpy as np
import cv2
import torch
import torch.backends.cudnn as cudnn
import io
import base64
import datetime
import argparse
import os
import sys
from pathlib import Path
import torch

import cv2
from PIL import Image
import time
from scipy.spatial import distance as dist
import argparse
import imutils
import time
import time
from threading import Thread
import math
import cv2
import playsound
import numpy as np
import threading
import cv2
import numpy as np
import pandas as pd
import csv
import numpy
from datetime import datetime

import math
ROOT = '/home/rcai/Desktop/yolov5'
if str(ROOT) not in sys.path:
    sys.path.append(str(ROOT))  # add ROOT to PATH
    ROOT = Path(os.path.relpath(ROOT, Path.cwd()))  # relative

    from utils.augmentations import letterbox 
    from models.common import DetectMultiBackend
    from utils.dataloaders import IMG_FORMATS, VID_FORMATS, LoadImages, LoadStreams
    from utils.general import (LOGGER, check_file, check_img_size, check_imshow, check_requirements, colorstr,
                           increment_path, non_max_suppression, print_args, scale_boxes, strip_optimizer, xyxy2xywh)
    from utils.plots import Annotator, colors, save_one_box
    from utils.torch_utils import select_device, time_sync
device = select_device('cpu')#Set 0 if you have GPU
model = DetectMultiBackend('D:\internship\yolov5-master\last.pt', device=device, dnn=False, data='data/coco128.yaml')
model.classes = [0, 2]
stride, names, pt, jit, onnx, engine = model.stride, model.names, model.pt, model.jit, model.onnx, model.engine
imgsz = check_img_size((640, 640), s=stride)  # check image size


dataset = LoadImages('D:\internship\yolov5-master\car.jpg', img_size=imgsz, stride=stride, auto=pt)

def draw_rect(image,points):
        x1=int(points[0])
        y1=int(points[1])
        x2=int(points[2])
        y2=int(points[3])
        midpoint=(int((x2+x1)/2),int((y2+y1)/2))
        print(midpoint)
    #print("Hi")
        cv2.rectangle(image, (x1,y1), (x2,y2), color = (255, 90, 90), thickness=4)
        cv2.circle(image, midpoint, radius=9, color=(0, 33, 45), thickness=-1)
        y_mid=int(y2+y1/2)
        return image, y_mid
def yolo(img):
        img0=img.copy()
        img = letterbox(img0, 640, stride=stride, auto=True)[0] 
        img = img.transpose((2, 0, 1))[::-1]  # HWC to CHW, BGR to RGB
        img = np.ascontiguousarray(img)
        im = torch.from_numpy(img).to(device)
        im = im.float() # uint8 to fp16/32
        im /= 255  # 0 - 255 to 0.0 - 1.0
        if len(im.shape) == 3:
            im = im[None]  # expand for batch dim
        dt = [0.0,0.0,0.0]    
        pred = model(im, augment=False, visualize=False)
        seen = 0
        pred = non_max_suppression(pred, conf_thres=0.45, iou_thres=0.45, classes=[0,1,2,3,4,6] , max_det=1000)
        det=pred[0]
        det[:, :4] = scale_boxes(im.shape[2:], det[:, :4], img0.shape).round()
        prediction=pred[0].cpu().numpy() 
        for i in range(prediction.shape[0]):
            imag,mid=draw_rect(img0,prediction[i,:])
        return imag,mid
def custom_infer(img0,
        weights='./best.pt',  # model.pt path(s),
        data='data/coco128.yaml',  # dataset.yaml path
        imgsz=(640, 640),  # inference size (height, width)
        conf_thres=0.35,  # confidence threshold
        iou_thres=0.45,  # NMS IOU threshold
        max_det=1000,  # maximum detections per image  # cuda device, i.e. 0 or 0,1,2,3 or cpu
        view_img=False,  # show results
        save_txt=False,  # save results to *.txt
        save_conf=False,  # save confidences in --save-txt labels
        save_crop=False,  # save cropped prediction boxes
        nosave=False,  # do not save images/videos
        classes=[0,1,2,3,4,6,8,10,12],  # filter by class: --class 0, or --class 0 2 3
        agnostic_nms=False,  # class-agnostic NMS
        augment=False,  # augmented inference
        visualize=False,  # visualize features
        update=False,  # update all models
        project=ROOT / 'runs/detect',  # save results to project/name
        name='exp',  # save results to project/name
        exist_ok=False,  # existing project/name ok, do not increment
        line_thickness=3,  # bounding box thickness (pixels)
        hide_labels=False,  # hide labels
        hide_conf=False,  # hide confidences
        half=False,  # use FP16 half-precision inference
        dnn=False,  # use OpenCV DNN for ONNX inference
        model=model):
        img = letterbox(img0, 640, stride=stride, auto=True)[0]

    # Convert
        img = img.transpose((2, 0, 1))[::-1]  # HWC to CHW, BGR to RGB
        img = np.ascontiguousarray(img)
        im = torch.from_numpy(img).to(device)
        im = im.float() # uint8 to fp16/32
        im /= 255  # 0 - 255 to 0.0 - 1.0
        if len(im.shape) == 3:
            im = im[None]  # expand for batch dim
        dt = [0.0,0.0,0.0]    
        pred = model(im, augment=augment, visualize=visualize)
        seen = 0
        if 1<2:
            pred = non_max_suppression(pred, conf_thres, iou_thres, classes, agnostic_nms, max_det=max_det)

        # Process predictions
            for i, det in enumerate(pred):  # per image
                seen += 1
                p, im0, frame = 'webcam.jpg', img0.copy(), getattr(dataset, 'frame', 0)

                p = Path(p)  # to Path
                imc = im0.copy() if save_crop else im0  # for save_crop
                annotator = Annotator(im0, line_width=line_thickness, example=str(names))
                if len(det):
                    # Rescale boxes from img_size to im0 size
                    det[:, :4] = scale_boxes(im.shape[2:], det[:, :4], im0.shape).round()

                # Print results
                    for c in det[:, -1].unique():
                        n = (det[:, -1] == c).sum()  # detections per class

                # Write results
                    for *xyxy, conf, cls in reversed(det):
                        if save_txt:  # Write to file
                            xywh = (xyxy2xywh(torch.tensor(xyxy).view(1, 4)) / gn).view(-1).tolist()  # normalized xywh
                            line = (cls, *xywh, conf) if save_conf else (cls, *xywh)  # label format
                            with open(txt_path + '.txt', 'a') as f:
                                f.write(('%g ' * len(line)).rstrip() % line + '\n')

                        if 1<2:  # Add bbox to image
                            c = int(cls)  # integer class
                            label = None if hide_labels else (names[c] if hide_conf else f'{names[c]} {conf:.2f}')
                            annotator.box_label(xyxy, label, color=colors(c, True))
                            if save_crop:
                                save_one_box(xyxy, imc, file=save_dir / 'crops' / names[c] / f'{p.stem}.jpg', BGR=True)

            # Stream results
                im0 = annotator.result()
        return im0,pred
from matplotlib import pyplot as plt

def my_resize(img):
    scale_percent = 10 # percent of original size
    width = int(img.shape[1] * scale_percent / 100)
    height = int(img.shape[0] * scale_percent / 100)
    dim = (width, height)
  
    # resize image
    resized = cv2.resize(img, dim, interpolation = cv2.INTER_AREA)
    return resized



# In[8]:
# import the opencv library
import cv2
import time
from datetime import datetime
import csv
import datetime



print("Im before while Loop")
window_name = "window"
vid = cv2.VideoCapture(0)
#cv2.namedWindow(window_name, cv2.WND_PROP_FULLSCREEN)
#cv2.setWindowProperty(window_name, cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN)

print("Im before while Loop 2" )
start=time.time()
while(True):
    print("Im inside the loop" )
    ret, frame = vid.read() 
    print("Frame is ",type(frame))
    print("Frame shape ",frame.shape)
    	
    
   
    # Image returned by yolo with classes
    pred_img = custom_infer(img0 = frame)[0]
    print(pred_img,"Predicted image") 
        
    #detected classes returned by Yolo
    detected_classes=custom_infer(img0 = frame)[1]
    detected_classes
    classses=detected_classes[:][0].cpu().numpy()[:,-1]
    print("Object undetected ",classses)
  
    local_time = datetime.datetime.now().strftime("%Y-%m-%d %I:%M:%S %p")

    det_cls=[]
    if len(classses) >0:
        print("Object detected")
        if classses==0:
            det_cls='yawning'
                    
        elif classses==1:
            det_cls='closed eyes'
                    
        elif classses==3:
            det_cls='mobile'
        print(det_cls)    
        print("local time:", local_time)
        list=[local_time , det_cls]        
    with open('car_det.csv', mode='a', newline='') as file:
       
        writer = csv.writer(file)
       
        writer.writerow(list)
        

            
    #cv2.imshow('ceh', pred_img)
    cv2.imshow("frame", pred_img)
    if cv2.waitKey(2) & 0xFF == ord('q'):
        break
vid.release()
cv2.destroyAllWindows()




vid = cv2.VideoCapture(0)
def main_page():
    def web_cam_func():
        def go_back_to_main_frame():
             cap.release()
             display_frame1.place_forget()
             display_frame2.place_forget()
             back_frame.place_forget()
             main_frame.place(relx=0.5, rely=0.5, width = 500, height = 500, anchor=tk.CENTER)
        
        main_frame.place_forget()
        width, height = 700, 700
        cap = cv2.VideoCapture(0)
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, width)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, height)

        display_frame1 = tk.Frame(root)
        display_frame1.place(relx=0.2, rely=0.5, width = 600, height = 700, anchor=tk.CENTER)

        display_frame1_label = tk.Label(display_frame1, text = "Original video", font = ('Rockwell', 16), bg = "yellow")
        display_frame1_label.pack(side=tk.TOP)

        display_frame2 = tk.Frame(root)
        display_frame2.place(relx=0.8, rely=0.5, width = 600, height = 700, anchor=tk.CENTER)

        display_frame2_label = tk.Label(display_frame2, text = "Detection", font = ('Rockwell', 16), bg = "yellow")
        display_frame2_label.pack(side=tk.TOP)

        back_frame = tk.Frame(root)
        back_frame.pack(side=tk.TOP, anchor=tk.NW)
        back_button = tk.Button(back_frame, text= "BACK", font=("Rockwell", 12), command=go_back_to_main_frame)
        back_button.pack()


        lmain = tk.Label(display_frame1)
        lmain1 = tk.Label(display_frame2)
        lmain.place(x = 0, y = 100, width=600, height=600)
        lmain1.place(x = 0, y = 100, width=600, height=600)
        
        def show_frame():
                _, frame = cap.read()
                frame2 = cv2.flip(frame, 1)
                cv2image = cv2.cvtColor(frame2, cv2.COLOR_BGR2RGBA)
                img = Image.fromarray(cv2image)

                imgtk = ImageTk.PhotoImage(image=img)
                lmain.imgtk = imgtk
                lmain.configure(image=imgtk)
                
                # Perform inference
                results = model(frame)

                 # Parse results and draw bounding boxes
                for *xyxy, conf, cls in results.xyxy[0]:
                    if conf>0.5:
                        label = f'{model.names[int(cls)]} {conf:.2f}'
                        cv2.rectangle(frame, (int(xyxy[0]), int(xyxy[1])), (int(xyxy[2]), int(xyxy[3])), (255,0,0), 2)
                        cv2.putText(frame, label, (int(xyxy[0]), int(xyxy[1])-10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255,0,0), 2)

                # frame3 = cv2.flip(frame, 1)
                frame3 = frame
                cv2image2 = cv2.cvtColor(frame3, cv2.COLOR_BGR2RGBA)
                img2 = Image.fromarray(cv2image2)

                imgtk2 = ImageTk.PhotoImage(image=img2)

                lmain1.imgtk = imgtk2
                lmain1.configure(image=imgtk2)
                
                lmain.after(10, show_frame)
            
        show_frame()
    def upload_vid_func():
        def browse_file():
            def run_yolov5_on_video():
                 def go_back_to_main_frame():
                    cap.release()
                    display_frame1.place_forget()
                    display_frame2.place_forget()
                    back_frame.place_forget()
                    main_frame.place(relx=0.5, rely=0.5, width = 500, height = 500, anchor=tk.CENTER)
                 
                 browse_frame.place_forget()
                 width, height = 700, 700
                #  print(file_path)
                 cap = cv2.VideoCapture(file_path)
                 cap.set(cv2.CAP_PROP_FRAME_WIDTH, width)
                 cap.set(cv2.CAP_PROP_FRAME_HEIGHT, height)

                 display_frame1 = tk.Frame(root)
                 display_frame1.place(relx=0.2, rely=0.5, width = 600, height = 700, anchor=tk.CENTER)

                 display_frame1_label = tk.Label(display_frame1, text = "Original video", font = ('Rockwell', 16), bg = "yellow")
                 display_frame1_label.pack(side=tk.TOP)

                 display_frame2 = tk.Frame(root)
                 display_frame2.place(relx=0.8, rely=0.5, width = 600, height = 700, anchor=tk.CENTER)

                 display_frame2_label = tk.Label(display_frame2, text = "Detection", font = ('Rockwell', 16), bg = "yellow")
                 display_frame2_label.pack(side=tk.TOP)

                 back_frame = tk.Frame(root)
                 back_frame.pack(side=tk.TOP, anchor=tk.NW)
                 back_button = tk.Button(back_frame, text= "BACK", font=("Rockwell", 12), command=go_back_to_main_frame)
                 back_button.pack()


                 lmain = tk.Label(display_frame1)
                 lmain1 = tk.Label(display_frame2)
                 lmain.place(x = 0, y = 100, width=600, height=600)
                 lmain1.place(x = 0, y = 100, width=600, height=600)
        
                 def show_frame():
                    
                    _, frame = cap.read()
                    # frame2 = cv2.flip(frame, 1)
                    frame2 = frame
                    cv2image = cv2.cvtColor(frame2, cv2.COLOR_BGR2RGBA)
                    img = Image.fromarray(cv2image)

                    imgtk = ImageTk.PhotoImage(image=img)
                    lmain.imgtk = imgtk
                    lmain.configure(image=imgtk)
                    
                    # Perform inference
                    results = model(frame)

                    # Parse results and draw bounding boxes
                    for *xyxy, conf, cls in results.xyxy[0]:
                        if conf>0.5:
                            label = f'{model.names[int(cls)]} {conf:.2f}'
                            cv2.rectangle(frame, (int(xyxy[0]), int(xyxy[1])), (int(xyxy[2]), int(xyxy[3])), (255,0,0), 2)
                            cv2.putText(frame, label, (int(xyxy[0]), int(xyxy[1])-10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255,0,0), 2)

                    # frame3 = cv2.flip(frame, 1)
                    frame3 = frame
                    cv2image2 = cv2.cvtColor(frame3, cv2.COLOR_BGR2RGBA)
                    img2 = Image.fromarray(cv2image2)

                    imgtk2 = ImageTk.PhotoImage(image=img2)

                    lmain1.imgtk = imgtk2
                    lmain1.configure(image=imgtk2)
                
                    lmain.after(1, show_frame)
                
                 show_frame()

            filename = filedialog.askopenfilename(filetypes=[("video files", "*.*")])
            file_path = os.path.abspath(filename)

            run_yolov5_on_video()
        
        main_frame.place_forget()

        browse_frame = tk.Frame(root, bg = "orange")
        browse_frame.place(relx=0.5, rely=0.5, anchor=tk.CENTER)
        
        browse_button = tk.Button(browse_frame, text="Browse", font= ("Rockwell", 20), bg="Yellow", fg="white", command=browse_file)
        browse_button.pack()

    main_frame = tk.Frame(root, bg="orange")

    main_frame.place(relx=0.5, rely=0.5, width = 500, height = 500, anchor=tk.CENTER)
    
    web_cam = tk.Button(main_frame, text = "Web cam", command = web_cam_func, bg = "yellow", fg = "purple", font=('Rockwell', 18))
    
    web_cam.place(x = 10, y = 100)
    
    upload_vid = tk.Button(main_frame, text = "Upload Video", command = upload_vid_func, bg = "yellow", fg = "purple", font=('Rockwell', 18))
    
    upload_vid.place(x = 300, y = 100)

main_page()

Title_label = tk.Label(root, text = "YOLOv5 Object detection", font = ('Rockwell', 20), bg = "yellow")
Title_label.pack(side=tk.TOP)

Exit_label = tk.Label(root, text = "Press excape to quit", font = ('Rockwell', 20), bg = "yellow")
Exit_label.pack(side=tk.BOTTOM)

# Execute tkinter
root.mainloop()