# escoliosis-app
# pasos
# funciona en python==3.9.0 y 3.6 
# pip install virtualenv
# para activar la virtual:
  # en cmd: escoliosis-app\venv\Scripts>activate.bat
# hay que instalar todo de nuevo 
  # pip install scikit-image
  # Successfully installed distlib-0.3.6 filelock-3.8.0 platformdirs-2.5.2 virtualenv-20.16.5

# luego en cmd: escoliosis-app>docker build -t scoliosisapp .
# y  escoliosis-app>docker run -it --publish 7000:4000 scoliosisapp

# y te dice: 
 # (venv) C:\Users\piotr\OneDrive\Escritorio\tfc version 3.1\escoliosis-app>docker run -it --publish 7000:4000 scoliosisapp
 #  * Serving Flask app 'app' (lazy loading)
 #  * Environment: production
 #    WARNING: This is a development server. Do not use it in a production deployment.
 #    Use a production WSGI server instead.
 #  * Debug mode: off
 #  * Running on all addresses.
 #    WARNING: This is a development server. Do not use it in a production deployment.
 #  * Running on http://172.17.0.2:4000/ (Press CTRL+C to quit)
 # INFO: Created TensorFlow Lite XNNPACK delegate for CPU.
 # 172.17.0.1 - - [25/Oct/2022 23:12:18] "POST /calcularAngulos HTTP/1.1" 200 -