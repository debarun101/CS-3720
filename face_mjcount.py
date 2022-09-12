# -*- coding: utf-8 -*-

import os
import sys
import time
import json
import logging
import argparse

import cv2
import numpy as np

import serial
import RPi.GPIO as GPIO



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


def main(args):
  # LEDs.
  GPIO.setmode(GPIO.BCM)
  GPIO.setup(args.LED_running, GPIO.OUT)
  GPIO.setup(args.LED_frame_processed, GPIO.OUT)

  GPIO.output(args.LED_running, True)

  with open(args.input_path, 'rb') as fp:
    faces_list = np.load(fp)

  maximum_count = majority_counts(faces_list)

  logging.info('Detected %i people in the video stream.', maximum_count[0])
  with open(args.output_path, 'w') as fp:
    results = {
      'number_of_people': maximum_count[0],
    }
    print >> fp, json.dumps(results, indent=2)

  GPIO.output(args.LED_running, False)
  GPIO.cleanup()

  fp = serial.Serial(args.serial_port, 9600)
  fp.write(str(maximum_count[0]) + '\n')
  fp.close()

if __name__ == '__main__':
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

  main(args)
  logging.info('Done')

  exit(0)
