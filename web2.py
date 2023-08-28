import tkinter as tk
from tkinter import filedialog
import cv2
import torch
from models.experimental import attempt_load
from utils.general import non_max_suppression

# Load your custom model from the 'last.pt' file
model = attempt_load('D:\internship\yolov5-master\last.pt')
# Set the device (cuda or cpu)
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Create the Tkinter GUI
root = tk.Tk()
root.title("Object Detection GUI")

# Create a Canvas widget to display video frames
canvas = tk.Canvas(root)
canvas.pack()
root.geometry("500x500")

def open_file():
    file_path = filedialog.askopenfilename()
    process_video(file_path)

def process_video(file_path):
    width, height = 700, 700
    cap = cv2.VideoCapture(file_path)

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        img = torch.from_numpy(frame).to(device)
        img = img.float() / 255.0
        img = img.permute(2, 0, 1).unsqueeze(0)

        with torch.no_grad():
            detections = model(img)[0]
            detections = non_max_suppression(detections, conf_thres=0.3, iou_thres=0.5)

        # Process and visualize detections on the frame
        # Your code for drawing bounding boxes and labels on the frame

        img_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        img_pil = Image.fromarray(img_rgb)
        img_tk = ImageTk.PhotoImage(image=img_pil)

        canvas.create_image(0, 0, anchor=tk.NW, image=img_tk)
        canvas.image = img_tk

        root.update_idletasks()
        root.update()

    cap.release()

open_button = tk.Button(root, text="Open Video", command=open_file)
open_button.pack()

root.mainloop()
