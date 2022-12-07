#pip install PyMuPDF
from doctest import ELLIPSIS_MARKER
from flask import Flask,send_file,request
import io
import matplotlib.pyplot as plt
import matplotlib.image as img
from skimage.io import imsave
from PIL import Image
import numpy as np
import math
import fitz
import pylab as pl
from matplotlib import collections  as mc
from mediapipe.python.solutions import drawing_utils as mp_drawing
from mediapipe.python.solutions import pose as mp_pose

scoliosisapp= Flask(__name__)

@scoliosisapp.route('/calcularAngulos', methods=['POST'])
def calculos():
    #si no se recibe la imagen     
    if 'imagefile' not in request.files:
        return 'No se recibió una imagen.'

    # Obtengo la imagen del request
    input_image = request.files['imagefile']

    # Convierto la imagen recibida a un byte array rgb
    input_image = np.array((Image.open(input_image)).convert('RGB'))

    # Inicializo el nuevo rastreador de poses para parte superior del cuerpo y ejecuto.
    upper_body_only=True
    with mp_pose.Pose(upper_body_only) as pose_tracker:
        result = pose_tracker.process(input_image)
        pose_landmarks = result.pose_landmarks
            
        # Si se reconocieron puntos de referencia.
        if pose_landmarks is not None:
            # Obtenemos los X e Y de los puntos de referencia
            pose_landmarks = [[lmk.x, lmk.y] for lmk in pose_landmarks.landmark]

            #Obtengo las dimensiones de la img recibida (alto x ancho)
            frame_height, frame_width = input_image.shape[:2]
            # Redimensiono los puntos al aspecto de la img recibida
            pose_landmarks *= np.array([frame_width, frame_height]) 

            #pose_landmarks = np.around(pose_landmarks, 5).flatten().astype(np.str).tolist()

            # Redondeo a 5 decimales (Investigar a cuantos decimales y porque)
            pose_landmarks = np.around(pose_landmarks, 5)

            #paso a diccionario            
            puntos_necesarios = {   0:pose_landmarks[11], 1:pose_landmarks[12],
                                    2:pose_landmarks[13], 3:pose_landmarks[14], 
                                    4:pose_landmarks[23], 5:pose_landmarks[24], 
                                }
        else:
            return "No se pudo obtener los puntos necesarios."
        
    #calculo la altura entre un punto y otro (y1-y2)
    cateto_opuesto_hombros = round(abs( float(puntos_necesarios[0][1]) - float(puntos_necesarios[1][1])),5)
    cateto_opuesto_codos =  round(abs(float(puntos_necesarios[2][1])- float(puntos_necesarios[3][1])),5)
    cateto_opuesto_cintura =  round(abs(float(puntos_necesarios[4][1]) - float(puntos_necesarios[5][1])),5)
    #calculo la distancia entre un punto y otro 
    hipot_hombros = np.sqrt((( float(puntos_necesarios[0][0])-float(puntos_necesarios[1][0]))**2)+(( float(puntos_necesarios[0][1])-float(puntos_necesarios[1][1]))**2))
    hipot_codos = np.sqrt((( float(puntos_necesarios[2][0])- float(puntos_necesarios[3][0]))**2)+(( float(puntos_necesarios[2][1])-float(puntos_necesarios[3][1]))**2))
    hipot_cintura = np.sqrt(((float(puntos_necesarios[4][0])-float(puntos_necesarios[5][0]))**2)+((float(puntos_necesarios[4][1])- float(puntos_necesarios[5][1]))**2))
    #calculo los angulos 
    angulo_hombro =round(math.degrees(math.asin(cateto_opuesto_hombros/hipot_hombros)),2)
    angulo_codo = round(math.degrees(math.asin(cateto_opuesto_codos/hipot_codos)),2)
    angulo_cintura = round(math.degrees(math.asin(cateto_opuesto_cintura/hipot_cintura)),2)

    angulos =   {   0:angulo_hombro, 
                    1:angulo_codo,
                    2:angulo_cintura,} 
    #calcular centros
    centro_hombro_x = (float(puntos_necesarios[0][0]) + float(puntos_necesarios[1][0]))/2
    centro_hombro_y = (float(puntos_necesarios[0][1]) + float(puntos_necesarios[1][1]))/2
    centro_codo_x = (float(puntos_necesarios[2][0]) + float(puntos_necesarios[3][0]))/2
    centro_codo_y = (float(puntos_necesarios[2][1]) + float(puntos_necesarios[3][1]))/2
    centro_cintura_x = (float(puntos_necesarios[4][0]) + float(puntos_necesarios[5][0]))/2
    centro_cintura_y = (float(puntos_necesarios[4][1]) + float(puntos_necesarios[5][1]))/2

    #genero el diagnostico
    tipo=0
    diagnosticos=[]
    tipos=[]
    for angulo in angulos:
        if angulo == 0:
            #"no posee"
            descripcion="no posee"
            tipo = 0
        elif angulo >= 4.1:
            descripcion="grave"
            tipo = 3
        elif angulo > 0 and angulo <= 1.5:
            descripcion="leve"
            tipo = 1
        elif angulo > 1.51 and angulo <= 4:
            descripcion="moderado" 
            tipo = 2
        diagnosticos.append(tipo)

    diagnosticoFinal=0
    for diag in diagnosticos:
        diagnosticoFinal=diagnosticoFinal+diag
    diagnosticoFinal=diagnosticoFinal/3
    if diagnosticoFinal == 0:
        descripcion="no posee"
    elif diagnosticoFinal >= 4.1:
        descripcion="grave"
    elif diagnosticoFinal > 0 and diagnosticoFinal <= 1.5:
        descripcion="leve"
    elif diagnosticoFinal > 1.51 and diagnosticoFinal <= 4:
        descripcion="moderado" 
    
    #separo en x e y para dibujar
    x_list = [float(puntos_necesarios[0][0]), float(puntos_necesarios[1][0]),
                float(puntos_necesarios[2][0]),float(puntos_necesarios[3][0]),
                float(puntos_necesarios[4][0]),float(puntos_necesarios[5][0]),
                centro_hombro_x, centro_codo_x, centro_cintura_x]

    y_list = [float(puntos_necesarios[0][1]), float(puntos_necesarios[1][1]),
                float(puntos_necesarios[2][1]),float(puntos_necesarios[3][1]),
                float(puntos_necesarios[4][1]),float(puntos_necesarios[5][1]),
                centro_hombro_y, centro_codo_y, centro_cintura_y]

    #genero las lineas 
    lines =[[(x_list[0], y_list[0]), (x_list[1], y_list[1])], 
            [(x_list[2], y_list[2]), (x_list[3], y_list[3])], 
            [(x_list[4], y_list[4]), (x_list[5], y_list[5])],
            [(x_list[6], y_list[6]), (x_list[6], y_list[8])]]

    lc = mc.LineCollection(lines)

    fig, ax = pl.subplots()
    x1=float(x_list[0])
    x2=float(x_list[2])
    x3= float(x_list[4])
    
    ang1=str(angulos[0])+'°'
    ang2=str(angulos[1])+'°'
    ang3= str(angulos[2])+'°'
    #agrego el label con los angulos 
    ax.text(x1, y_list[0], ang1, color='red',fontweight ='bold',bbox ={'facecolor':'white','alpha':0.5, 'pad':1})
    ax.text(x2, y_list[2], ang2, color='red',fontweight ='bold',bbox ={'facecolor':'white','alpha':0.5, 'pad':1})
    ax.text(x3, y_list[4], ang3, color='red',fontweight ='bold',bbox ={'facecolor':'white','alpha':0.5, 'pad':1})
    plt.imshow(input_image)
    ax.set_title('Resultados')
    ax.add_collection(lc)   
    ax.autoscale()
    #plt.show()
    strIO = io.BytesIO()
    plt.savefig(strIO, dpi=fig.dpi)
    strIO.seek(0)

    # Nuevo documento
    doc = fitz.open()
    # Nueva página en el documento. Se insertará tras la última página
    pagina = doc.new_page(pno=-1,width=1240,height=1754)
    # Establecemos la posición sobre la que vamos a dibujar
    posicion = fitz.Point(100, 200)
    posicion2 = fitz.Point(200, 300)
    # Insertamos un texto en la página
    pagina.insert_text(posicion, "Pre-diagnostico obtenido:", fontsize=50)
    pagina.insert_text(posicion2, str(descripcion), fontsize=50)
    #insertamos imagen
    input_image = request.files['imagefile']
    pagina.insert_image(rect=(365, 360, 765, 860),stream=strIO, keep_proportion=True, overlay=True)
    # Guardamos los cambios en el documento
    doc.write()
    # Guardamos el fichero PDF
    doc.save("prueba.pdf", pretty=True)

    #return send_file(strIO, mimetype='image/png')

#https://evilnapsis.com/2019/03/28/crear-pdf-agregar-imagenes-y-parrafos-con-reportlab-en-python/
#@scoliosisapp.route('/imprimirDiagnostico', methods=['POST'])
#def imprimirDiagnostico():
    # # Nuevo documento
    # doc = fitz.open()
    # # Nueva página en el documento. Se insertará tras la última página
    # pagina = doc.new_page(pno=-1,width=1240,height=1754)
    # # Establecemos la posición sobre la que vamos a dibujar
    # posicion = fitz.Point(100, 200)
    # # Insertamos un texto en la página
    # pagina.insert_text(posicion, "¡Hola PyMuPDF!", fontsize=50)
    # #insertamos imagen
    # input_image = request.files['imagefile']
    # pagina.insert_image(rect=(365, 360, 765, 860),stream=input_image, keep_proportion=True, overlay=True)
    # # Guardamos los cambios en el documento
    # doc.write()
    # # Guardamos el fichero PDF
    # doc.save("prueba.pdf", pretty=True)
#     # Nuevo documento
#     doc = fitz.open()
#     # Nueva página en el documento. Se insertará tras la última página
#     pagina = doc.new_page(pno=-1,width=1240,height=1754)
#     # Establecemos la posición sobre la que vamos a dibujar
#     posicion = fitz.Point(100, 200)
#     # Insertamos un titulo en la página
#     pagina.insert_text(fitz.Point(170,100), "Pre-diagnostico obtenido", color=1, fontsize=30)

# #     # Obtengo la imagen del request
# #     #input_image = request.files['imagefile']
   
# #     #insertamos la img
# #     # El tamaño de la gráfica es de 12 * dpi + 9 * dpi
# #     #pagina.insert_image((150,130,150+(12*80),130+(9*80)), stream=buf, keep_proportion=True)
#     # Insertamos los cambios en el documento
#     doc.write()
#     # Guardamos el archivo PDF
#     doc.save("prueba.pdf", pretty=True)
    

if __name__== '__main__':
    scoliosisapp.run(host="0.0.0.0", port=4000)
