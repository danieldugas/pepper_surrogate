sudo apt install -y libhidapi-dev

# udev rules
sudo apt install -y python3-attr
cd ~/Code
git clone https://gitlab.freedesktop.org/monado/utilities/xr-hardware.git
cd ~/Code/xr-hardware
make
python3 make-udev-rules.py > 70-xrhardware.rules
sudo cp 70-xrhardware.rules /etc/udev/rules.d/

# pepper kinematics
source ~/peppervenv/bin/activate
cd ~/Code/pepper_ws/src/pepper_surrogate/external/python_pepper_kinematics
pip install -e .
pip install scipy

# cv2
source ~/peppervenv/bin/activate
pip install opencv-python==4.2.0.32

