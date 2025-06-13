"""
/*
 *-------------------------------------------------------------------------------------------------------------------------------------->
 *	Name: RetinaFace_Live_RK3588.py
 *-------------------------------------------------------------------------------------------------------------------------------------->
 *	Purpose: Retinaface Model Live Camera for RK3588
 *-------------------------------------------------------------------------------------------------------------------------------------->
 *	Dependent Reference
 *-------------------------------------------------------------------------------------------------------------------------------------->
 *	Known Issues
 *-------------------------------------------------------------------------------------------------------------------------------------->
 *	Methodology
 *-------------------------------------------------------------------------------------------------------------------------------------->
 *	References
 *-------------------------------------------------------------------------------------------------------------------------------------->
 *	MSDN documents
 *-------------------------------------------------------------------------------------------------------------------------------------->
 *	Internal documents
 *-------------------------------------------------------------------------------------------------------------------------------------->
 *	Internet documents
 *-------------------------------------------------------------------------------------------------------------------------------------->
 *  (01)RKNN Toolkit2
 *      https://github.com/airockchip/rknn-toolkit2/
 *-------------------------------------------------------------------------------------------------------------------------------------->
 *  (02)RKNN Model Zoo
 *      https://github.com/airockchip/rknn_model_zoo
 *--------------------------------------------------------------------------------------------------------------------------------------> 
*/
"""

# Import Module
import cv2
import time
import numpy as np
from rknnlite.api import RKNNLite
from RetinaFace_Post_Process import funDecodeBoxes, funDecodeLandmarks, funGeneratePriorities, funNMS

MODEL_PATH = './RetinaFace.rknn'

# Initialize RKNN Model
rknn = RKNNLite()
rknn.load_rknn(MODEL_PATH)
rknn.init_runtime()

# Open Camera
# If you didn't not Camera Index, please run: ./ListCamera.sh to find it
cap = cv2.VideoCapture(11)
# Configure Camera Resoultion 320x320 (RetinaFace.rknn only accepts this size)
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 320)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 320)

# Configure RetinaFace.rknn Input Size, related parameters
# Height, Width
input_size = (320, 320)
# Minimum size of anchor for each layer
min_sizes = [[16, 32], [64, 128], [256, 512]]
# Characteristics of each layer stride
steps = [8, 16, 32]
variances = [0.1, 0.2]
conf_threshold = 0.7

# Pre-generated All Priority Frames (anchors)
priors = funGeneratePriorities(input_size, min_sizes, steps)

while True:
    ret, frame = cap.read()
    if not ret:
        break

    # Convert image size to model input size
    H, W = input_size
    frame_resized = cv2.resize(frame, (W, H))
    # Convert BGR to RGB to float32
    img_rgb = cv2.cvtColor(frame_resized, cv2.COLOR_BGR2RGB).astype(np.float32)
    # Increate batch dimension
    img_rgb = img_rgb[np.newaxis, :]

    t0 = time.time()
    # Performing Model Inference
    outputs = rknn.inference(inputs=[img_rgb])
    t1 = time.time()

    if outputs is None:
        print("Inference failed, check input format or model compatibility.")
        break

    # Get model output (Prediction Box, Confidence, Key Points)
    loc = np.array(outputs[0]).reshape(-1, 4)
    conf = np.array(outputs[1]).reshape(-1, 2)
    landms = np.array(outputs[2]).reshape(-1, 10)

    # Filtering Faces above a confidence threshold
    scores = conf[:, 1]
    inds = np.where(scores > conf_threshold)[0]
    if inds.size == 0:
        # No face detected, update FPS, time to next frame
        fps = 1 / (time.time() - t0)
        infer_time = (t1 - t0) * 1000
        cv2.putText(frame_resized, f'FPS: {fps:.2f}', (10, 20),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
        cv2.putText(frame_resized, f'Detect: {infer_time:.2f} ms', (10, 45),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
        cv2.imshow('RetinaFace Live', frame_resized)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
        continue

    # Decode the actual boxes and Key Points based on the filtered anchors
    boxes = funDecodeBoxes(loc[inds], priors[inds], variances)
    landms = funDecodeLandmarks(landms[inds], priors[inds], variances)
    scores = scores[inds]

    # Merge boxes and scores and perform NMS filtering on overlapping boxes
    dets = np.hstack((boxes, scores[:, np.newaxis])).astype(np.float32)
    keep = funNMS(dets)
    dets = dets[keep]
    landms = landms[keep]

    # Draw Face Detection Box, Face Key Points
    for b, lm in zip(dets, landms):
        x1 = int(b[0] * W);  y1 = int(b[1] * H)
        x2 = int(b[2] * W);  y2 = int(b[3] * H)
        cv2.rectangle(frame_resized, (x1, y1), (x2, y2), (0, 255, 0), 2)
        for i in range(5):
            cx = int(lm[2 * i] * W);  cy = int(lm[2 * i + 1] * H)
            cv2.circle(frame_resized, (cx, cy), 2, (0, 0, 255), -1)

    # Display FPS, Inference Milliseconds
    fps = 1 / (time.time() - t0)
    infer_time = (t1 - t0) * 1000
    cv2.putText(frame_resized, f'FPS: {fps:.2f}', (10, 20),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
    cv2.putText(frame_resized, f'Detect: {infer_time:.2f} ms', (10, 45),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)

    cv2.imshow('RetinaFace Live', frame_resized)
    # Wait for Keyboard Input q to exit infinity loop
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release Resources
cap.release()
cv2.destroyAllWindows()
rknn.release()
