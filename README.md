# DeepRes-Video-Summarization
A Deep Learning-based Video Summarization Strategy for Resource-Constrained Industrial Surveillance Scenarios

## Paper
https://ieeexplore.ieee.org/document/8815938

# Repository Explaination
There are total three Python script files and two required folders [Coarse-refine, Candidate-keyframes] to run the code in this repository (Explained alphabetically).

### 1. coarse_refining.py

It takes a video as an input, apply the methodology given in Section II.B [Coarse Refining] and writes the the coarse refined frames in the relevant "Coarse-refine" folder.

### 2. fine_refining.py

It processes the output frames from the first Python script, apply matching distance technique over the coarse-refined frames, as explained in Section II.C [Fine Refining] of the paper. This script finally outputs a series of candidate keyframes in "Candidate-keyframes" folder.

### 3. keyframes_generation.py

This code is borrowed from:
- https://github.com/tanveer-hussain/Embedded-Vision-for-MVS
and readers are suggested to tune their thresholds as per their requirements, and to generate skims of an input video, one can directly consider the candidate keyframes.


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


