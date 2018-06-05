# How to install COMMIT_debugger on Ubuntu

COMMIT_debugger is tested on:
- cmake version 3.5.1
- make version GNU Make 4.1
- Ubuntu version 14.04 and 16.04


1) Packeges to be installed:

sudo apt-get install libglu1-mesa-dev freeglut3-dev mesa-common-dev libxmu-dev libxi-dev libboost-program-options-dev libnifti-dev libblitz0-dev cmake


2) COMMIT works on g++-4.9:

sudo apt-get install g++-4.9

Use symbolic link to change the global version:
cd /usr/bin
sudo rm g++ cpp
sudo ln -s g++-4.9 g++
sudo ln -s cpp-4.9 cpp

--------------------
gcc -version 5.4
g++ —version 4.9.3
cpp —version 4.9.3
c++ --version 4.9.3
-------------------


3) Use it on Ubuntu:

git clone https://github.com/barakovic/COMMIT.git COMMIT_Barakovic
cd COMMIT_Barakovic
git checkout Ubuntu/COMMIT_debugger
cd extras
mkdir build
cd build
cmake ..
sudo make install

4) if error in blitz:
  remove libblitz0-dev and install blitzpp-blitz-1.0.1-0-g19079b6.tar.gz (in the files of COMMIT)


5) Copy path in .bashrc


6) if segmentation fault then run following command before:
export LD_PRELOAD=/usr/lib/x86_64-linux-gnu/mesa/libGL.so.1
