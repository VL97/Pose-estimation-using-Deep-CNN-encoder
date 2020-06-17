# Pose-estimation-using-Deep-CNN-encoder

Basic Keras model for the proposed Network in paper <a href=https://arxiv.org/pdf/1312.4659v3.pdf >DeepPose: Human Pose Estimation via Deep Neural Networks. </a>

10k images from <a href=http://human-pose.mpi-inf.mpg.de/ > MPII dataset </a> were used for training and full body images from <a href=https://bensapp.github.io/flic-dataset.html> FLIC dataset </a> images were used for visual performance testing. No performance benchmarks are evaluated as of now. Simple 1-pass non-cascading architecture was used.

MPII dataset matlab annotations file was preprocessed using https://github.com/asanakoy/deeppose_tf/blob/master/datasets/mpii_dataset.py
to obtain cleaned datampii.json.

Some test results:
<table>
 <tr>
  <td> <img src=https://github.com/VL97/Pose-estimation-using-Deep-CNN-encoder/blob/master/testresult/hitch-00163191.jpg > </img> </td>
  <td> <img src=https://github.com/VL97/Pose-estimation-using-Deep-CNN-encoder/blob/master/testresult/giant-side-a-00083861.jpg > </img> </td>
  <td> <img src=https://github.com/VL97/Pose-estimation-using-Deep-CNN-encoder/blob/master/testresult/hitch-00023501.jpg > </img> </td>
  <td> <img src=https://github.com/VL97/Pose-estimation-using-Deep-CNN-encoder/blob/master/testresult/million-dollar-baby-disc-00055321.jpg > </img> </td>
 </tr>
 
  <tr>
  <td> <img src=https://github.com/VL97/Pose-estimation-using-Deep-CNN-encoder/blob/master/testresult/the-departed-00205831.jpg > </img> </td>
  <td> <img src=https://github.com/VL97/Pose-estimation-using-Deep-CNN-encoder/blob/master/testresult/ten-commandments-disc1-00038441.jpg > </img> </td>
  <td> <img src=https://github.com/VL97/Pose-estimation-using-Deep-CNN-encoder/blob/master/testresult/schindlers-list-00128601.jpg > </img> </td>
  <td> <img src=https://github.com/VL97/Pose-estimation-using-Deep-CNN-encoder/blob/master/testresult/monster-in-law-d1-00016011.jpg > </img> </td>
 </tr>
