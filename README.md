# PatchMatchFusionNetwork
Improved implementation of PatchMatch that utilizes fusion network

# Instructions to access code:

Download DFNet folder from fusion network repository, found below

Download, open, and run "PatchMatchX.py" in that folder

To use different parameters, scroll to bottom:

Replace "./samples/test/place3.jpg" with desired image name

in call to blackRectangle method, change x and y positions and width and height of hole by changing the 4 values

Change patch size with p_size

Change number of iterations with itr

Run non-multi-scaled PatchMatch with NNS, multi-scaled PatchMatch with MSNNS, or neural-network-informed-multi-scaled-PatchMatch with MSNNNNS

The call to the network is in the line "os.system('python test.py --model model/model_places2.pth --img samples/test/img --mask samples/test/mask --output output/test --merge')"

A different PyTorch model can be used instead of mode_places2.pth

Different masks can be used by changing the mask path

The other paths are preset by PatchMatchX.py

Original PatchMatch python implementation by MingtaoGuo, edited for the purposes of this project: github.com/MingtaoGuo/PatchMatch

Original fusion network by Mr.Blue, edited for the purposes of this project: github.com/hughplay/DFNet

