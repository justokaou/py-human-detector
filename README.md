# Human Detector

This project utilizes the pre-trained YOLOv3 model. In my Python project, I am utilizing this model for real-time human detection, as part of potential future surveillance camera applications.

## Before Getting Started

You will need to create a `.env` file as follows:
```
user = [your sender email]
password = [your email password]

dest = [email of the recipient who will be notified]
```

## Installation

To run this project, you need to install the required libraries. You can do this by executing the following command :

```bash
pip install -r requirements.txt
```

## Usage

### Desktop environment

To run the script in a desktop environment, you can use :

```bash
python3 main.py
```

### Cli environment

If you don't have access to a desktop environment, you can use :

```bash
xvfb-run python3 main.python3
```

#### Alternatively

You can edit and use the provided `start.sh` script.

If you are using a Python virtual environment :

```bash
#!/bin/bash

while true
do
echo -e start human detector

source [path-to-your-venv]/venv/bin/activate
xvfb-run python3 ./main.py
deactivate

echo -e restarting human detector in 5 seconds
sleep 5
done
```

If you are not using a Python virtual environment :
```bash
#!/bin/bash

while true
do

echo -e start human detector
xvfb-run python3 ./main.py

echo -e restarting human detector in 5 seconds
sleep 5
done
```

Afterwards, make sure to run the following command :

```bash
chmod +x start.sh
```
Now you can run the project using :

```bash
./start.sh
```
