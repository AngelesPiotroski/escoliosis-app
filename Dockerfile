FROM python:3.9

WORKDIR /app
COPY . /app

ENV DEBIAN_FRONTEND noninteractive
RUN apt-get update -y
RUN apt install libgl1-mesa-glx -y
RUN apt-get install 'ffmpeg'\
    'libsm6'\
    'libxext6'  -y
RUN pip3 install --upgrade pip

RUN pip3 install opencv-python
RUN pip3 install imageio
RUN pip3 install scikit-image  
RUN pip3 install numpy 
RUN pip3 install matplotlib 
RUN pip3 install mediapipe 
RUN pip3 install Flask 
RUN pip3 install fitz
RUN pip3 install PyMuPDF
RUN pip3 install iteration_utilities

CMD ["python3","./src/app.py"]
