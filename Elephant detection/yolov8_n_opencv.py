# import random
# import serial
# import time
# import cv2
# import numpy as np
# from ultralytics import YOLO
# from playsound import playsound
# import winsound

# # opening the file in read mode
# my_file = open("utils/coco.txt", "r")
# # reading the file
# data = my_file.read()
# # replacing end splitting the text | when newline ('\n') is seen.
# class_list = data.split("\n")
# my_file.close()

# # Generate random colors for class list
# detection_colors = []
# for i in range(len(class_list)):
#     r = random.randint(0, 255)
#     g = random.randint(0, 255)
#     b = random.randint(0, 255)
#     detection_colors.append((b, g, r))

# # load a pretrained YOLOv8n model
# model = YOLO("best.pt", "v8")

# # Vals to resize video frames | small frame optimise the run
# frame_wid = 640
# frame_hyt = 480

# #cap = cv2.VideoCapture(0)
# cap = cv2.VideoCapture("test-video.m4v")

# if not cap.isOpened():
#     print("Cannot open camera")
#     exit()

# v = 'off'
# sound_playing = False  # Initialize sound_playing variable

# def play_sound():
#     winsound.PlaySound("many-bees-flying-around-27383.wav", winsound.SND_ASYNC + winsound.SND_LOOP)

# def stop():
#     winsound.PlaySound(None, 0)

# elephant_detected = False
# elephant_detection_start_time = None

# while True:
#     # Capture frame-by-frame
#     ret, frame = cap.read()
#     # if frame is read correctly ret is True

#     if not ret:
#         print("Can't receive frame (stream end?). Exiting ...")
#         break

#     #  resize the frame | small frame optimise the run
#     # frame = cv2.resize(frame, (frame_wid, frame_hyt))

#     # Predict on image
#     detect_params = model.predict(source=[frame], conf=0.55, save=False)

#     # Convert tensor array to numpy
#     DP = detect_params[0].numpy()
#     if len(DP) != 0:
#         v = 'on'
#         for i in range(len(detect_params[0])):
#             boxes = detect_params[0].boxes
#             box = boxes[i]  # returns one box
#             clsID = box.cls.numpy()[0]
#             conf = box.conf.numpy()[0]
#             bb = box.xyxy.numpy()[0]

#             cv2.rectangle(
#                 frame,
#                 (int(bb[0]), int(bb[1])),
#                 (int(bb[2]), int(bb[3])),
#                 detection_colors[int(clsID)],
#                 3,
#             )

#             # Display class name and confidence
#             font = cv2.FONT_HERSHEY_COMPLEX
#             cv2.putText(
#                 frame,
#                 class_list[int(clsID)] + " " + str(round(conf, 3)) + "%",
#                 (int(bb[0]), int(bb[1]) - 10),
#                 font,
#                 1,
#                 (255, 255, 255),
#                 2,
#             )

#             if "elephant" in class_list[int(clsID)].lower():
#                 if not elephant_detected:
#                     elephant_detection_start_time = time.time()
#                 elephant_detected = True
#                 if time.time() - elephant_detection_start_time >= 5:
#                     if not sound_playing:
#                         play_sound()  # Start playing the sound
#                         sound_playing = True
#             else:
#                 elephant_detected = False
#     else:
#         v = 'off'
#         elephant_detected = False

#     # s.write(v.encode())
#     # print(s.readline().decode('ascii'))
#     # Display the resulting frame
#     cv2.imshow("ObjectDetection", frame)

#     # Terminate run when "Q" pressed
#     if cv2.waitKey(1) == ord("q"):
#         break

# # When everything done, release the capture
# cap.release()
# cv2.destroyAllWindows()
# # s.close()


import random
import time
import cv2
import winsound
import numpy as np
from ultralytics import YOLO

# Read class labels from the file
with open("utils/coco.txt", "r") as my_file:
    class_list = my_file.read().splitlines()

# Generate random colors for each class
detection_colors = [
    (random.randint(0, 255), random.randint(0, 255), random.randint(0, 255))
    for _ in range(len(class_list))
]

# Load the YOLO model
model = YOLO("best.pt", "v8")

# Video frame dimensions
frame_wid = 640
frame_hyt = 480

# Open the video or camera
cap = cv2.VideoCapture("test-video.m4v")
if not cap.isOpened():
    print("Cannot open video source")
    exit()

# Sound control
def play_sound():
    winsound.PlaySound("many-bees-flying-around-27383.wav", winsound.SND_ASYNC | winsound.SND_LOOP)

def stop_sound():
    winsound.PlaySound(None, 0)

# Detection flags and timing
elephant_detected = False
elephant_detection_start_time = None
sound_playing = False

while True:
    # Capture frame-by-frame
    ret, frame = cap.read()
    if not ret:
        print("Video stream ended. Exiting...")
        break

    # Predict objects in the frame
    results = model.predict(source=[frame], conf=0.55, save=False)

    # Process detections
    for detection in results[0].boxes:
        cls_id = int(detection.cls.numpy()[0])
        confidence = detection.conf.numpy()[0]
        bbox = detection.xyxy.numpy()[0]

        # Draw bounding box and label
        cv2.rectangle(
            frame,
            (int(bbox[0]), int(bbox[1])),
            (int(bbox[2]), int(bbox[3])),
            detection_colors[cls_id],
            3
        )
        label = f"{class_list[cls_id]} {confidence:.2f}%"
        cv2.putText(frame, label, (int(bbox[0]), int(bbox[1]) - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)

        # Elephant detection logic
        if "elephant" in class_list[cls_id].lower():
            if not elephant_detected:
                elephant_detection_start_time = time.time()
            elephant_detected = True
            if time.time() - elephant_detection_start_time >= 5 and not sound_playing:
                play_sound()
                sound_playing = True
        else:
            elephant_detected = False

    # Stop sound if no elephant detected
    if not elephant_detected and sound_playing:
        stop_sound()
        sound_playing = False

    # Display the frame
    cv2.imshow("Object Detection", frame)

    # Exit on 'q' key press
    if cv2.waitKey(1) == ord("q"):
        break

# Release resources
cap.release()
cv2.destroyAllWindows()
stop_sound()
