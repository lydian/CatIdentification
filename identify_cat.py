import argparse
import json
import time
from datetime import datetime
from collections import Counter

import cv2
import numpy as np
import requests
from tflite_runtime.interpreter import Interpreter


model_path = "./model.tflite"
classes_path = "./cls.txt"
img_size = 300


def fetch_and_resize_image(video_path):
    cap = cv2.VideoCapture(video_path)
    (success, frame) = cap.read()
    if not success:
        return
    frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    frame = cv2.resize(frame, (img_size, img_size), interpolation = cv2.INTER_AREA)
    return (frame / 255.).astype("float32").reshape(1, img_size, img_size, 3)


def fetch_frames(video_path, time_limit):
    start_time_in_seconds = time.time()
    count = 0
    frames = []
    while time.time() - start_time_in_seconds < time_limit:
        frame = fetch_and_resize_image(video_path)
        if frame is None:
            break
        frames.append(frame)
        count += 1
    return frames

def set_input_tensor(interpreter, image):
    tensor_index = interpreter.get_input_details()[0]['index']
    input_tensor = interpreter.tensor(tensor_index)()[0]
    input_tensor[:, :] = image

def predict(interpreter, predict_to_cls, image):
    set_input_tensor(interpreter, image)
    interpreter.invoke()
    output_details = interpreter.get_output_details()
    prediction = np.argmax(interpreter.get_tensor(output_details[0]['index']))
    return predict_to_cls[prediction] 


def send_ifttt(ifttt_url, start, prediction):
    if prediction in ("boba", "hojicha"):
        msg = f"{prediction} is in the litter box"
    elif prediction == "rotate":
        msg = "litter robot is self-cleaning"
    else:
        msg = "some one is cleaning the litter robot"
    return requests.post(ifttt_url, data={"value1": prediction, "value2": start, "value3": msg})

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--ifttt", type=str, help="provide ifttt url if you want to send notification", default=None)
    parser.add_argument("--time", type=int, help="how many seconds of images you want to fetch for the prediction", default=10)
    parser.add_argument("video_url", help="url to fetch videos")

    return parser.parse_args()


def run():
    start = time.time()
    args = parse_args()
    print(f"fetching {args.time} seconds of images")
    frames = fetch_frames(args.video_url, args.time)
    print(f"fetched {len(frames)}")
    with open(classes_path) as fp:
        predict_to_cls = {int(k): v for k, v in json.load(fp).items()}
    
    interpreter = Interpreter(model_path)
    interpreter.allocate_tensors()
    
    start = datetime.now()
    counter = Counter([predict(interpreter, predict_to_cls, frame) for frame in frames])
    prediction = counter.most_common(1)[0][0]
    print(f"predict: {prediction} using {len(frames)} data within {datetime.now() -start}")
    if args.ifttt is not None:
        send_ifttt(args.ifttt, start, prediction)


if __name__ == "__main__":
    run()
