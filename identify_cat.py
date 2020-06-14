import argparse
import json
import time
from collections import Counter

import cv2
import numpy as np
import requests
import tensorflow as tf


model_path = "./model"
classes_path = "./model/cls.txt"


def fetch_and_resize_image(video_path):
    cap = cv2.VideoCapture(video_path)
    (success, frame) = cap.read()
    if not success:
        return
    frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    frame = cv2.resize(frame, (300, 300), interpolation = cv2.INTER_AREA)
    return frame


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


def predict(images):
    model = tf.keras.models.load_model(model_path)
    with open(classes_path) as fp:
        predict_to_cls = {int(k): v for k, v in json.load(fp).items()}
    images = images / 255.
    predictions = model.predict(images)
    c = Counter(np.argmax(predictions, axis=1))
    return predict_to_cls[c.most_common(1)[0][0]]


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
    frames = np.stack(fetch_frames(args.video_url, args.time))
    prediction = predict(frames)
    print(f"predict: {prediction} using {frames.shape} data")
    if args.ifttt is not None:
        send_ifttt(args.ifttt, start, prediction)


if __name__ == "__main__":
    run()
