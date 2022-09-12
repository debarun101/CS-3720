#!/bin/sh

set -x

sleep_secs=30

while [ 1 -eq 1 ]; do
  # Processs face representation.
  python "apps/face_reps_orig.py" \
    --output_path="output/faces.npz" \
    || exit -1

  python "apps/face_mjcount.py" \
    --input_path="output/faces.npz" \
    --output_path="output/result.json" \
    || exit -1
  
#  # Clusters face.
#  python "apps/face_clusters.py" \
#    --input_path="output/faces.npz" \
#    --output_path="output/result.json" \
#    || exit -1
  
  # Send data via serial port.
  count=`grep -o -E "[0-9]+" "output/result.json"`
  python "apps/serial_send.py" \
    --number_of_people="${count}" \
    || exit -1

  # Sleeps.
  sleep ${sleep_secs}
done


exit 0
