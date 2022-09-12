# -*- coding: utf-8 -*- 

import os
import sys
import time
import json
import logging
import argparse
import traceback
import datetime

import cv2
import numpy as np
from picamera import PiCamera
from picamera.array import PiRGBArray

import dlib
import openface

import RPi.GPIO as GPIO
import time
import serial
#from subprocess import call

#set GPIO Pins
GPIO_TRIGGER = 3
GPIO_ECHO = 4
#set GPIO direction (IN / OUT)
GPIO.setmode(GPIO.BCM)
GPIO.setup(GPIO_TRIGGER, GPIO.OUT)
GPIO.setup(GPIO_ECHO, GPIO.IN)


def majority_counts(faces_list):
    counts_map = {}
    maximum = ('', 0)
    for frame_id, faces in enumerate(faces_list):
        n = len(faces['faces'])
        print('len-faces:' + str(n))
	if n in counts_map: counts_map[n] += 1
	else: counts_map[n] = 1

	if counts_map[n] > maximum[1]: maximum = (n, counts_map[n])
	
    print(str(maximum[0]))
    return maximum


def face_counts():
    logging.basicConfig(level=logging.DEBUG)
    
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '--input_path',
        type=str,
        help='Path to the input npz file storing face embedding vectors.',
        default='output/faces.npz')
    parser.add_argument('--imgDim', type=int, default=96)
    parser.add_argument('--distance_threshold', type=float, default=0.2)
    parser.add_argument(
        '--output_path',
        type=str,
        help='Path to the output json file.',
        default='output/result.json')
    
    parser.add_argument('--LED_running', type=int, default=21)
    parser.add_argument('--LED_frame_processed', type=int, default=20)
    parser.add_argument('--serial_port', type=str, default='/dev/ttyACM0')
    
    args = parser.parse_args()
    logging.info(json.dumps(vars(args), indent=2))
    
    with open(args.input_path, 'rb') as fp:
        faces_list = np.load(fp)
    
    maximum_count = majority_counts(faces_list)
    
    logging.info('Detected %i people in the video stream.', maximum_count[0])
    with open(args.output_path, 'w') as fp:
        results = {
            'number_of_people': maximum_count[0],
            }
        print >> fp, json.dumps(results, indent=2)
    
    fp = serial.Serial(args.serial_port, 9600)
    fp.write(str(maximum_count[0]) + '\n')
    fp.close()
    return maximum_count
    

def distance():
	# set Trigger to HIGH
        GPIO.output(GPIO_TRIGGER, True)
   
      	# set Trigger after 0.01ms to LOW
      	time.sleep(0.00001)
     	GPIO.output(GPIO_TRIGGER, False)
   
      	StartTime = time.time()
      	StopTime = time.time()
   
      	# save StartTime
      	while GPIO.input(GPIO_ECHO) == 0:
		StartTime = time.time()
   
      	# save time of arrival
     	while GPIO.input(GPIO_ECHO) == 1:
		StopTime = time.time()
   
      	# time difference between start and arrival
      	TimeElapsed = StopTime - StartTime
        # multiply with the sonic speed (34300 cm/s)
        # and divide by 2, because there and back
        distance = (TimeElapsed * 34300) / 2
           
	return distance

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
    

def serial_send(nbr):
    parser = argparse.ArgumentParser()
    parser.add_argument('--number_of_people', type=int, default=nbr)
    parser.add_argument('--device', type=str, default='/dev/ttyACM0')

    args = parser.parse_args()
    logging.info(json.dumps(vars(args), indent=2))

    count = args.number_of_people
    with serial.Serial(args.device, 9600) as fp:
        fp.write(str(count) + '\n')
    logging.info('Number of people: {}'.format(count))
    logging.info('Done')


def main(args):
  # Initialize face detector.
  # LEDs.
  GPIO.setmode(GPIO.BCM)
  GPIO.setup(args.LED_running, GPIO.OUT)
  GPIO.setup(args.LED_frame_processed, GPIO.OUT)
  GPIO.output(args.LED_running, True)
  
  #Create Log File
  #f = open("count_log.txt","a+")
  ace_cascade = None
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
  
  try:
      while True:
          #f = open("count_log.txt","a+")
          dist = distance()
          print ("Measured Distance = %.1f cm" % dist)
          
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
          
          if dist < 50:
              
              print ("Measured Distance = %.1f cm" % dist)
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
              nbr_face = face_counts()
              #update log file
              f = open("/home/pi/Work/iot/apps/count_log.txt","a+")
              now = datetime.datetime.now()
              timedate = now.strftime("%Y-%m-%d,%H:%M")
              f.write(str(nbr_face[0]))
              f.write(",")
              f.write(timedate)
              f.write("\n")
              f.close()
              #send number of faces to arduino
              serial_send(nbr_face)
              #call(["python", "apps/face_mjcount.py", "--input_path='output/faces.npz'", "--output_path='output/result.json'"])
              
           
          GPIO.output(args.LED_running, False)
          
       #   GPIO.cleanup()
          time.sleep(0.3)          
  # Reset by pressing CTRL + C
  except KeyboardInterrupt:
      print("Measurement stopped by User")
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
