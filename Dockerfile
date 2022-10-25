FROM python:3.6.8

WORKDIR /app
COPY . /app

ENV DEBIAN_FRONTEND noninteractive
RUN apt-get update -y
RUN apt install libgl1-mesa-glx -y
RUN apt-get install 'ffmpeg'\
    'libsm6'\
    'libxext6'  -y
RUN pip3 install --upgrade pip

RUN pip3 install opencv-python==4.3.0.38
RUN pip3 install imageio
RUN pip3 install scikit-image  
RUN pip3 install numpy 
RUN pip3 install matplotlib 
RUN pip3 install mediapipe 
RUN pip3 install Flask 

EXPOSE 80
CMD ["python3","./src/app.py"]
