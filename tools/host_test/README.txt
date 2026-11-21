Optional PC test of the firmware maths (Linux or WSL, needs g++). No board required.

  ./run_host_test.sh /path/to/Chirale_TensorFLowLite

Part 1 compiles firmware/hushnet_poc/hn_features.h for the PC and compares the int8 features of the
replay clips with the ones NumPy computed in the notebook (stored in hn_replay_clips.h).

Part 2 compiles the TensorFlow Lite Micro sources of the Arduino library for the PC, links them with
firmware/hushnet_poc/hn_inference.h and compares the int8 scores with the ones from the notebook.

Result for the files shipped in this folder:
  features: 0 of 1960 values differ, on all 8 clips
  scores:   largest difference 0, on all 8 clips; 7 of 8 clips correct, same as the notebook
  arena:    28288 bytes used (64 bit PC)

Run it again after you retrain: it tells you in a minute whether the new headers and the firmware
agree, before you flash anything.
