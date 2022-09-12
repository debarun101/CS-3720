# -*- coding: utf-8 -*- 

import os
import sys
import time
import json
import logging
import argparse

import cv2
import numpy as np
from picamera import PiCamera
from picamera.array import PiRGBArray

import dlib
import openface

import RPi.GPIO as GPIO


def frame_loop(process_func, global_status, height=240, width=320, frame_rate=20):
  """Reads frames from PiCamera, processes bgr frames.

  Args:
    process_func: a callable that takes [height, width, 3] bgr frame as input.
    global_status: a dict storing global status.
    height: height of the captured frame.
    width: width of the captured frame.
    frame_rate: number of frames captures within one second.
  """
  with PiCamera() as camera:
    camera.resolution = (width, height)
    camera.framerate = frame_rate
    rawCapture = PiRGBArray(camera, size=(width, height))
    
    # 2 sec warmup 
    time.sleep(2)
    
    # capture frames from the camera
    for frame in camera.capture_continuous(
        rawCapture, format='bgr', use_video_port=True):

      if process_func(frame.array):
        break
    
      # clear the stream in preparation for the next frame
      rawCapture.truncate(0)

    
def update_global_status(global_status, faces):
  """Updates global status.

  Args:
    global_status: a dict storing global status.
    faces: detected faces in the current frame.
  """
#  if global_status['number_of_frames'] == 0:
#    # TODO(yek@): check the reason why the first frame is empty.
#    faces = []

  global_status['faces'].append(faces)
  global_status['number_of_frames'] += 1

  now = time.time()
  duration = now - global_status['checkpoint']['check_time']
  if duration >= 1.0:
    frames_processed = global_status['number_of_frames'] \
                       - global_status['checkpoint']['number_of_frames']

    global_status['frame_rate'] = frames_processed / duration
    global_status['checkpoint']['check_time'] = now
    global_status['checkpoint']['number_of_frames'] += frames_processed
    logging.info('#frames=%i, #faces=%i', 
        global_status['number_of_frames'], len(faces))


def detect_faces(align, face_cascade, rgb_frame):
  """Detects faces in the video frame.

  Args:
    align: an openface.AlignDlib instance.
    face_cascade: a cv2.CascadeClassifier instance.
    rgb_frame: [height, width, 3] rgb image.
  """
  # OpenCV face detection, faster.
  if face_cascade is not None:
    gray = cv2.cvtColor(rgb_frame, cv2.COLOR_RGB2GRAY)
    faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=3)

  # Openface face detection.
  else:
    bb = align.getAllFaceBoundingBoxes(rgb_frame)
    faces = [(box.left(), box.top(), box.width(), box.height()) for box in bb]

  return faces


def align_faces(align, rgb_frame, boxes, img_dim):
  """Align faces.

  Args:
    align: an openface.AlignDlib instance.
    rgb_frame: [height, width, 3] rgb image.
    boxes: a list of (x, y, w, h) tuple.
    img_dim: size of the face thumbnail.
  """
  aligned_faces = []
  for box in boxes:
    x, y, w, h = box
    aligned_faces.append(
        align.align(img_dim, rgb_frame,
          dlib.rectangle(x, y, x + w, y + h),
          landmarkIndices=openface.AlignDlib.OUTER_EYES_AND_NOSE)
        )
  return aligned_faces


def represent_faces(net, faces_list):
  """Represents and clusters faces.

  Args:
    faces_list: a list of list of faces.

  Returns:
    faces: a list of faces after clustering.
  """
  reps_list = []
  for frame_id, faces in enumerate(faces_list):
    logging.info('Processing frame {}.'.format(frame_id))
    reps = [{'face': face, 'rep': net.forward(face)} for face in faces]
    reps_list.append({
        'frame_id': frame_id,
        'faces': reps})
  return reps_list


def process_frame(bgr_frame, align, face_cascade, global_status, args):
  """Processes a video frame.

  Args:
    frame: a [height, width, 3] rgb frame.
    align: an openface.AlignDlib instance.
    face_cascade: a cv2.CascadeClassifier instance.
    global_status: a dict storing global status.
    args: the program arguments.

  Returns:
    stop: if True, stop processing.
  """
  rgb_frame = cv2.cvtColor(bgr_frame, cv2.COLOR_BGR2RGB)

  # Face detection.
  boxes = detect_faces(align, face_cascade, rgb_frame)

  # Face alignment.
  aligned_faces = align_faces(align, rgb_frame, boxes, args.imgDim)

  # Update global_status.
  update_global_status(global_status, aligned_faces)

  GPIO.output(args.LED_frame_processed, True)
  time.sleep(0.1)
  GPIO.output(args.LED_frame_processed, False)

  # Visualize face detection and alignment.
  for (x, y, w, h) in boxes:
    cv2.rectangle(bgr_frame, (x, y), (x + w, y + h), (0, 255, 0), 2)

  aligned_faces = [cv2.cvtColor(face, cv2.COLOR_RGB2BGR) \
                  for face in aligned_faces]

  cv2.imshow("LiveView", bgr_frame)
  if len(aligned_faces) > 0:
    cv2.imshow("Thumbnails", np.concatenate(aligned_faces, 1))

  key = cv2.waitKey(500) & 0xFF

  if global_status['number_of_frames'] > args.number_of_frames:
    return True

  return False
    

def main(args):
  # LEDs.
  GPIO.setmode(GPIO.BCM)
  GPIO.setup(args.LED_running, GPIO.OUT)
  GPIO.setup(args.LED_frame_processed, GPIO.OUT)

  GPIO.output(args.LED_running, True)

  # Initialize face detector.
  face_cascade = None
  if args.use_opencv_face_detector:
    assert os.path.isfile(args.face_cascade_model_name)
    face_cascade = cv2.CascadeClassifier(args.face_cascade_model_name)

  assert os.path.isfile(args.dlibFacePredictor)
  align = openface.AlignDlib(args.dlibFacePredictor)

  assert os.path.isfile(args.networkModel)
  net = openface.TorchNeuralNet( 
      args.networkModel,
      imgDim=args.imgDim,
      cuda=False)

  # Global status.
  global_status = {
    'width': args.width,
    'height': args.height,
    'frame_rate': 0.0,
    'number_of_frames': 0,
    'checkpoint': {
      'check_time': time.time(),
      'number_of_frames': 0,
    },
    'faces': [],
  }

  # Windows.
  cv2.namedWindow('LiveView')
  cv2.namedWindow('Thumbnails')

  # Frame loop.
  process_func = lambda frame: process_frame(
      frame, align, face_cascade, global_status, args)

  frame_loop(process_func, 
      global_status,
      height=args.height, 
      width=args.width,
      frame_rate=args.frame_rate)

  # Face representation.
  faces = represent_faces(net, global_status['faces'])
  with open(args.output_path, 'wb') as fp:
    np.save(fp, faces)

  cv2.destroyAllWindows()

  GPIO.output(args.LED_running, False)
  GPIO.cleanup()


if __name__ == '__main__':
  logging.basicConfig(level=logging.DEBUG)

  parser = argparse.ArgumentParser()
  parser.add_argument(
      '--face_cascade_model_name', 
      type=str,
      help='Path to the cascade classification model.',
      default='configs/haarcascade_frontalface_default.xml')
  parser.add_argument( 
      '--dlibFacePredictor',
      type=str,
      help="Path to dlib's face predictor.",
      default='openface/models/dlib/shape_predictor_68_face_landmarks.dat')
  parser.add_argument(
      '--networkModel',
      type=str,
      help="Path to Torch network model.",
      default='openface/models/openface/nn4.small2.v1.t7')
  parser.add_argument(
      '--output_path', 
      type=str, 
      help='Path to the output npz file storing face embedding vectors.',
      default='output/faces.npz')
  parser.add_argument('--width', type=int, default=1600)
  parser.add_argument('--height', type=int, default=900)
  parser.add_argument('--frame_rate', type=int, default=5)
  parser.add_argument('--imgDim', type=int, default=96)
  parser.add_argument('--use_opencv_face_detector', type=int, default=1)
  parser.add_argument('--number_of_frames', type=int, default=4)

  parser.add_argument('--LED_running', type=int, default=21)
  parser.add_argument('--LED_frame_processed', type=int, default=20)

  args = parser.parse_args()
  logging.info(json.dumps(vars(args), indent=2))

  main(args)
  logging.info('Done')

  exit(0)
