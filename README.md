# Defectron
Defect detection framework on cylindrical metal components on robotic cell using computer vision technologies (deep learning models).

## Dependencies
- Python 3.9
- zmqRemoteApi
- Torch 1.10.1
  ```batch
  pip install torch==1.10.1+cpu torchvision==0.11.2+cpu torchaudio==0.10.1 -f https://download.pytorch.org/whl/cpu/torch_stable.html
  ```
- tqdm
- opencv-python
- scipy
- open3d
- scikit-image
- filterpy
- pandas

## Installation
1- Install CoppeliaSim Edu from [here](https://www.coppeliarobotics.com/downloads.html)
2- Install all dependencies above using the following command:
  ```batch
  pip install -r requirements.txt
  ```
> **Note**
> you also need to install yolov7 requirements in the `yolov7` folder.

3- Install zmqRemoteApi (CoppeliaSim API) from [here](https://github.com/CoppeliaRobotics/zmqRemoteApi.git)
4- Download supporting files and models from [here](https://drive.google.com/file/d/1nUyOOTvOTajbFAaojP_rhz6TH5PhsBEh/view?usp=share_link) and put YoloV7 model folder inside the `src` folder. 
> **Warning**
> Don't change the folder name of the YoloV7 model folder.
> It should look like this `src/yolov7_model`.

## Run
1- To run this program make sure you installed all dependencies and supporting files and models.
2- Run CoppeliaSim Edu and open the scene file `scene.ttt` in the `data/cell` folder.
> **Note**
> Don't start the scene from CoppeliaSim, it will be started automatically from the program.

3- Run the main entry points which is `defectron.py` file in the `main` folder.
4- Make sure you change the workpiece information in the `defectron.py` file to match your workpiece, if you are using a new workpiece different from the one in the scene.
5- When you run the `defectron.py` file and the scene is started, make sure you speed up the simulation to its maximum speed by pressing the "bunny" button in the top right corner of the CoppeliaSim window.
6- Now, relax and enjoy the show. :smile:
> **Note**
> If the run fails from the first time, try to run it again as YoloV7 might require to download the onnx package.

## Run Sample
![sample1]()

