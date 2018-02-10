#!/bin/sh

apt-get update
apt-get install python3-pip python3-tk
pip3 install tensorflow-gpu pillow h5py matplotlib imageio sklearn scipy keras
