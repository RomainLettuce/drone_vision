# Drone Vision
This repository is for localizing a drone's gps coordinates from top view videos.
It is implemented based on the grpc to serve video streaming from the server.

## Setup - Server

1. Clone git repo

>     cd ~
>     git clone https://github.com/RomainLettuce/drone_vision.git
>     cd drone_vision/server

2. Create a conda environment (or venv)

>     conda create -n drone_env python=3.10
>     conda activate drone_env

3. Install requirements

>     pip install -r requirements.txt

4. Install grpc tools

>     pip install grpcio grpcio-tools

5. Build proto file

>     cp ../proto/drone.proto .
>     python -m grpc_tools.protoc \
>       --proto_path=. \
>       --python_out=. \
>       --grpc_python_out=. \
>       drone.proto

## Setup - Client
1. Clone the repository

>     cd ~
>     git clone https://github.com/RomainLettuce/drone_vision.git
>     cd drone_vision/client

2. Create a conda environment (or venv)

>     conda create -n drone_env python=3.10
>     conda activate drone_env

3. Install requirements

>     pip install -r requirements.txt

5. Install grpc tools

>     pip install grpcio grpcio-tools

6. Build proto file

>     cp ../proto/drone.proto .
>     python -m grpc_tools.protoc \
>       --proto_path=. \
>       --python_out=. \
>       --grpc_python_out=. \
>       drone.proto

## How to use?
### Server
Execute `server.py` with proper options. There are several options you can set.
options:
 - \-\-video /path/to/target/video
 - \-\-cones-gps /path/to/cones/gps
 - \-\-output-csv /path/to/save/output
 - \-\-start-time The first timestamp in the SRT file
 - \-\-host Host ip (default: 0.0.0.0)
 - \-\-port Port # to use (default: 50051)
 - \-\-cone-spacing-m distance between cones (in meters) for the 24 cones case
 - \-\-cone-spacing-vertical-m vertical distance between two cones (in meters) for the 4 cones case
 - \-\-cone-spacing-horizontal-m horizontal disance ,,,
 - \-\-refA refernce cone A ({row#}_{col#}) (4 cones case)
 - \-\-refB reference cone B ({row#}_{col#}) (4 cones case)
 - \-\-cones-layout {24, 4}

Example (24 cones case):

    python server.py --video ~/drone_vision_temp/test.mp4 --cones-gps ../cone-gps/24_cones.json --output-csv test.csv --start-time "2025-10-23 15:25:58.334" --cones-layout 24

Example (4 cones case):

    python server.py --video ./DJI_20251028105919_0019_D.MP4 --cones-gps 2x2_gps_1028.json --output-csv test.csv --start-time "2025-05-20 11:38:43.161" --cones-layout 4 --refA 0_0 --refB 1_0

### Client
Execute `client.py` with correct server ip and port number.

Example:

    python client.py --server 111.222.333.444:50051

### Usage
1. After execute `client.py`, a window for video stream will be presented (it takes about a minute for 24 cones case because it calculate best reference cones for the first stage). 
2. If a initial frame of the video is presented, press `r` button to select ROI (region of interest.)
3. Drag a ROI box and press `Enter`.
	Tip) If resolution (image quality) is good, set a tight ROI box, eliminating backgrounds from the box as much. Otherwise, include some background in the ROI box.
4. You should monitor the tracking output because the tracker is somtimes bound with unwanted region. In this case, you have to press `r` button as soon as possible to set a ROI box again.
5. Also, if the tracker lost its target, it will stop automatically, and you have to press `r` button to set a ROI box again.
6. If the target escape the monitor, press `q` button to finish the processing and save a csv file.
7. If you want to check the output, convert the csv to kml file at this link: https://www.convertcsv.com/csv-to-kml.htm. Then, check the position by uploading the kml file to the google earth project.
