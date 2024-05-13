# ARTgram + Hyperscanning With Muse EEG
This repository contains information for measuring inter-brain synchrony using the MUSE EEG system while participants jointly participate in an AR Tangram task. The simultaneous measurement of neural responses from collaborating partners is known as hyperscanning. More specifically, it provides:
* Instructions on how to configure a computer to conduct hyperscanning with Muse.
* Code for analyzing the resulting data.

# Hyperscanning Dependencies
1. PC computer
2. EEG System: [Muse 2 Headset](https://choosemuse.com/muse-2/)
3. Lab Streaming Layer: [BlueMuse](https://github.com/kowalej/BlueMuse)
4. Muse LSL: [muselsl](https://github.com/alexandrebarachant/muse-lsl/)
5. Audio Mixer: [VB-Audio](https://vb-audio.com/Voicemeeter/banana.htm)
6. BIDS Data Format: [LabRecorder](https://github.com/labstreaminglayer/App-LabRecorder)
7. ARTgram: Augmented Reality Tangram Task 

# How To Administer The Task In Your Lab

See [artgram_muse_procedure.pdf](https://github.com/cogcommscience-lab/muse_artgram/blob/main/artgram_muse_procedure.pdf)

# Analysis Code
* Step 1: Reads in hyperscanning data, cleans it, and structures it for later analyses
* Step 2: Structure EEG data with self-report and behavioral data
* Step 3: Conduct a multiverse analysis to understand the impact of different cleaning choices on your data
* Step 4: Conduct specific analyses based on parameters selected from the multiverse analysis
* Step 5: Calculate how much data is lost to autoreject

# Raw Data
The raw EEG data are too large to upload to GitHub. They are stored on OSF at:

All other raw data are uploaded as .csv files.

