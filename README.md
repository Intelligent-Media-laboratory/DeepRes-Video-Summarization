DeepReS: A Deep Learning-based Video Summarization Strategy for Resource-Constrained Industrial Surveillance Scenarios
=======

## Paper
https://ieeexplore.ieee.org/document/8815938

## Requirements
- python v >= 3.5
- sklearn
- cv2 v >= 3.4.1
- numpy
- caffe
- scipy
- glob, os

## How to install the requirements and run the code
The Python implementation of this paper has some prerequisites that are necassary for running the code.
The two most important dependencies are: OpenCV >= 3.4.1 and Caffe (deep learning framework).

OpenCV can be easily installed by following the below links:
- https://jeanvitor.com/cpp-opencv-windonws10-installing/
- https://docs.opencv.org/3.4.3/d3/d52/tutorial_windows_install.html

These are links for pre-built OpenCV libraries.

Caffe building is a little bit hard and you need to follow up a straight forward tutorial from YouTube. 

Here is the official caffe installation page, which you may find hard to follow, but do not worry I will also share a tutorial link
- https://caffe.berkeleyvision.org/installation.html
- https://github.com/BVLC/caffe/tree/windows

Here is the YouTube link:

- https://www.youtube.com/watch?v=nrzAF2sxHHM

Furthermore, you need to install scikit-learn, scipy, and numpy which are easy to install and can be done through "pip" command.

Once the requirements are installed, you can easily execute the codes, that are explained below.

# Repository Explaination
There are total three Python script files and two required folders [Coarse-refine, Candidate-keyframes] to run the code in this repository (Explained alphabetically). 
##### The folders contain "coarse refined" and "candidate keyframes" generated by all the compared CNN models with the given thresholds as in the research paper (see Table IV caption).

### 1. coarse_refining.py

It takes a video as an input, apply the methodology given in Section II.B [Coarse Refining] and writes the the coarse refined frames in the relevant "Coarse-refine" folder.

### 2. fine_refining.py

It processes the output frames from the first Python script, apply matching distance technique over the coarse-refined frames, as explained in Section II.C [Fine Refining] of the paper. This script finally outputs a series of candidate keyframes in "Candidate-keyframes" folder.

### 3. keyframes_generation.py

This code is borrowed from:
https://github.com/tanveer-hussain/Embedded-Vision-for-MVS
and readers are suggested to tune their thresholds as per their requirements, and to generate skims of an input video, a user can directly consider the candidate keyframes.


# Citation
<pre>
<code>
The paper is accepted and soon it will be online.
</code>
</pre>

If you are interested in Video Summarization domain you may find some of my other recent papers worthy to read:

<pre>
<code>
K. Muhammad, T. Hussain, and S. W. Baik, "Efficient CNN based summarization of surveillance videos for resource-constrained devices," Pattern Recognition Letters, 2018/08/07/ 2018

Hussain, Tanveer, Khan Muhammad, Amin Ullah, Zehong Cao, Sung Wook Baik, and Victor Hugo C. de Albuquerque. "Cloud-Assisted Multi-View Video Summarization using CNN and Bi-Directional LSTM." IEEE Transactions on Industrial Informatics (2019).

K. Muhammad, T. Hussain, M. Tanveer, G. Sannino and V. H. C. de Albuquerque, "Cost-Effective Video Summarization using Deep CNN with Hierarchical Weighted Fusion for IoT Surveillance Networks," in IEEE Internet of Things Journal.
doi: 10.1109/JIOT.2019.2950469
</code>
</pre>


