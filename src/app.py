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
import re
import pylab as pl
from matplotlib import collections  as mc
from mediapipe.python.solutions import drawing_utils as mp_drawing
from mediapipe.python.solutions import pose as mp_pose
from iteration_utilities import duplicates,unique_everseen

scoliosisapp= Flask(__name__)

@scoliosisapp.route('/imprimirPdf', methods=['POST'])
def obtenerDiagnostico():
    #si no se recibe la imagen     
    if 'imagefile' not in request.files:
        return 'No se recibió una imagen.'
    # Obtengo la imagen del request
    input_image = request.files['imagefile']
    # Convierto la imagen recibida a un byte array rgb
    input_image = np.array((Image.open(input_image)).convert('RGB'))
    # Envio la imagen para ser procesada y obtengo los puntos
    puntos_necesarios = obtenerPuntos(input_image)
    # Envio los puntos para realizar los calculos y obtengo los resultados
    datos = calcularDiagnostico(puntos_necesarios)
    # Envio los resultados para generar el pdf
    generarPdf(puntos_necesarios, input_image, datos)
    # Retorno el archivo PDF
    return send_file("nuevo.pdf", mimetype='application/pdf')

def obtenerPuntos(input_image):
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
            puntos_necesarios = {   #hombros izq y der:
                                    0:pose_landmarks[11], 1:pose_landmarks[12],
                                    #codos izq y der:
                                    2:pose_landmarks[13], 3:pose_landmarks[14], 
                                    #cintura izq y der:
                                    4:pose_landmarks[23], 5:pose_landmarks[24], 
                                    #orejas izq y der:
                                    6:pose_landmarks[7], 7:pose_landmarks[8], 
                                }
            return(puntos_necesarios)
        else:
            return "No se pudo obtener los puntos necesarios."

def calcularDiagnostico(puntos_necesarios):
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
    #centro de las orejas (occipital)
    centro_oreja_x = (float(puntos_necesarios[6][0]) + float(puntos_necesarios[7][0]))/2
    centro_oreja_y = (float(puntos_necesarios[6][1]) + float(puntos_necesarios[7][1]))/2

    #triangulo de la talla
    distancia_codo_izq = abs(centro_oreja_x - float(puntos_necesarios[2][0]))
    distancia_codo_der = abs(centro_oreja_x - float(puntos_necesarios[3][0]))
    if distancia_codo_der > distancia_codo_izq :
        triangulo_talla = "se detecta el triángulo de la talla a la izquierda."
        triangulo =True
    elif distancia_codo_der < distancia_codo_izq:
        triangulo_talla = "se detecta el triángulo de la talla a la derecha."
        triangulo = True
    if distancia_codo_izq == distancia_codo_der:
        triangulo = False
        triangulo_talla = "no se detectó el triángulo de la talla."

    #verificacion de la postura equilibrada: 
    hombro_equilibrado = abs(centro_oreja_x - centro_hombro_x)
    codos_equilibrado = abs(centro_oreja_x - centro_codo_x)
    cintura_equilibrado = abs(centro_oreja_x - centro_cintura_x)
    #genero el diagnostico de la postura: (equilibrada- desequilibrada)
    postura_equilibrada=""
    if hombro_equilibrado <= 0.5 and codos_equilibrado <= 0.5 and cintura_equilibrado <= 0.5 : 
        postura_equilibrada = "Si"
    else: 
        postura_equilibrada = "No"
   
    #el desequilibrio es hacia la derecha o izquierda?????
    desequilibrio= ""
    if hombro_equilibrado <= 0.5 and codos_equilibrado <= 0.5 and cintura_equilibrado > 0.5 :
        desequilibrio = "Posterior"
    elif hombro_equilibrado > 0.5 and codos_equilibrado <= 0.5 and cintura_equilibrado <= 0.5 :
        desequilibrio = "Superior"
    elif hombro_equilibrado < 0.5 and codos_equilibrado >= 0.5 and cintura_equilibrado < 0.5 :
        desequilibrio = "Media"
    elif  hombro_equilibrado > 0.5 and codos_equilibrado > 0.5 and cintura_equilibrado > 0.5 :
        desequilibrio = "Posterior y superior"

    #genero el diagnostico de escoliosis
    tipo=0
    diagnosticos=[]
    descripciones=[]
    aux=[angulos[0],angulos[1],angulos[2]]
    for angulo in aux:
        if angulo == 0 or angulo <= 0.5:
            #"no posee"
            tipo = 0
            descripcion="No posee escoliosis"
        elif angulo >= 4.01:
            #"grave"
            tipo = 3
            descripcion="Escoliosis grave"
        elif angulo > 0.5 and angulo <= 1.5:
            #"leve"
            tipo = 1
            descripcion="Escoliosis leve"
        elif angulo > 1.51 and angulo <= 4:
            #"moderado" 
            tipo = 2
            descripcion="Escoliosis moderado"
        diagnosticos.append(tipo) 
        descripciones.append(descripcion)

    #si hay duplicados va a estar el duplicado y sino significa que todos son distintos y por ende sera moderado
    diag = list(unique_everseen(duplicates(descripciones)))
    if len(diag) == 0:
        promedio="Escoliosis moderado"
        diag.append(promedio)
    
    datos={ 0:centro_hombro_x, 1:centro_codo_x, 2:centro_cintura_x,
            3:centro_oreja_x, 4:centro_hombro_y, 5:centro_codo_y,
            6:centro_cintura_y, 7:centro_oreja_y, 8:triangulo_talla,
            9:triangulo, 10:angulos, 11:diag, 12:postura_equilibrada,
            13:desequilibrio, 14:descripciones
        }
    return datos

def generarPdf(puntos_necesarios,input_image,datos):
    #separo en x e y para dibujar
    x_list = [float(puntos_necesarios[0][0]), float(puntos_necesarios[1][0]),
                float(puntos_necesarios[2][0]),float(puntos_necesarios[3][0]),
                float(puntos_necesarios[4][0]),float(puntos_necesarios[5][0]),
                datos[0], datos[1], datos[2],datos[3]]

    y_list = [float(puntos_necesarios[0][1]), float(puntos_necesarios[1][1]),
                float(puntos_necesarios[2][1]),float(puntos_necesarios[3][1]),
                float(puntos_necesarios[4][1]),float(puntos_necesarios[5][1]),
                datos[4], datos[5], datos[6],datos[7]]

    #genero las lineas 
    lines =[[(x_list[0], y_list[0]), (x_list[1], y_list[1])], 
            [(x_list[2], y_list[2]), (x_list[3], y_list[3])], 
            [(x_list[4], y_list[4]), (x_list[5], y_list[5])],
            [(x_list[9], y_list[9]), (x_list[9], y_list[8])]]
    if "derecha" in datos[8]:
        #EL TRIANGULO ESTA A LA DERECHA
        # esta ultima seria: desde centro_oreja_x y centro_oreja_y hasta el centro_oreja_x y centro_cintura_y
        lineas_triangulo = [[(x_list[1], y_list[0]), (datos[1], datos[5])], 
                            [(x_list[5], y_list[4]), (datos[1], datos[5])], 
                            [(x_list[1], y_list[0]), (x_list[1], y_list[5])]]
        #para dibujar el triangulo de la talla
        #poner una linea de xy del hombro, hasta xy de la cintura, desde xy del centro_codos hasta xy de la cintura y desde xy del centro_codos hasta xy del hombro
    elif "izquierda" in datos[8]:
        #EL TRIANGULO ESTA A LA IZQUIERDA
        # esta ultima seria: desde centro_oreja_x y centro_oreja_y hasta el centro_oreja_x y centro_cintura_y
        lineas_triangulo = [[(x_list[0], y_list[0]), (datos[1], datos[5])], 
                            [(x_list[4], y_list[4]), (datos[1], datos[5])], 
                            [(x_list[0], y_list[0]), (x_list[0], y_list[4])]]

    lc = mc.LineCollection(lines)

    fig, ax = pl.subplots()
    x1=float(x_list[0])
    x2=float(x_list[2])
    x3= float(x_list[4])
    #figura 1 con las lineas y angulos
    ang1=str(datos[10][0])+'°'
    ang2=str(datos[10][1])+'°'
    ang3= str(datos[10][2])+'°'
    #agrego el label con los angulos 
    ax.text(x1, y_list[0], ang1, color='red',fontweight ='bold',bbox ={'facecolor':'white','alpha':0.5, 'pad':1})
    ax.text(x2, y_list[2], ang2, color='red',fontweight ='bold',bbox ={'facecolor':'white','alpha':0.5, 'pad':1})
    ax.text(x3, y_list[4], ang3, color='red',fontweight ='bold',bbox ={'facecolor':'white','alpha':0.5, 'pad':1})
    plt.imshow(input_image)
    plt.grid(True)
    plt.tight_layout()
    ax.set_title('Imágen 1')
    ax.add_collection(lc)   
    ax.autoscale()
    #plt.show()
    imagenAngulos = io.BytesIO()
    plt.savefig(imagenAngulos, dpi=fig.dpi)
    imagenAngulos.seek(0)

    #creamos la figura 2 con el triangulo de la talla
    ltriangulo = mc.LineCollection(lineas_triangulo)

    fig2, ax2 = pl.subplots()
    plt.imshow(input_image)
    ax2.set_title('Imágen 2: Triángulo de la talla')
    ax2.add_collection(ltriangulo)   
    ax2.autoscale()
    #plt.show()
    imagenTriangulo = io.BytesIO()
    plt.savefig(imagenTriangulo, dpi=fig2.dpi)
    imagenTriangulo.seek(0)

    
    listaResultados= list(datos[14])
    resultadoHombro= re.sub('Escoliosis',"",listaResultados[0] )
    resultadoCodo= re.sub('Escoliosis',"",listaResultados[1])
    resultadoCintura= re.sub('Escoliosis',"",listaResultados[2]) 
   
    # Nuevo documento
    doc = fitz.open()
    # Nueva página en el documento. Se insertará tras la última página
    pagina = doc.new_page(pno=-1,width=1240,height=1754)
    # Insertamos un texto en la página
    pagina.insert_text(fitz.Point(150, 100), "Pre-diagnóstico obtenido: "+str(datos[11][0]), fontsize=50)
    
    pagina.insert_text(fitz.Point(50, 150), "El pre-diagnóstico indicado se obtiene promediando los resultados de cada uno de los ángulos que se muestran en la Imágen 1.\n \n"+
    "Los resultados cada uno de los ángulos detectados en su fotografía son: \n"+
     " - En los hombros: "+resultadoHombro+ "\n"+
     " - En los codos: "+resultadoCodo+ "\n"+
     " - En la cintura: "+resultadoCintura +"\n"+

    "Estos resultados estan basados en los siguientes rangos: \n"+
    " * Si posee un ángulo entre: 0.5 - 1.5 grados, la escoliosis será: Leve \n"+    
    " * Si posee un ángulo entre: 1.51 - 4 grados, la escoliosis será: Moderado \n"+  
    " * Si posee un ángulo mayor o igual a 4.1 grados. la escoliosis será: Grave. \n"+
    "El resultado final que representa al pre-diagnóstico visualizado corresponde al promedio entre: \n "+
                        "( "+resultadoHombro+" + "+resultadoCodo+" + "+resultadoCintura+" ) % 3 = "+datos[11][0]+ "\n", fontsize=20)
    #insertamos imagen 1 
    pagina.insert_image((165, 550,1065, 1160),stream=imagenAngulos, keep_proportion=True)
    #SI EXISTE TRIANGULO LO GRAFICO Y MUESTRO, SINO SOLO SE DICE QUE NO SE DETECTO
    if datos[9] == True:
        pagina.insert_text(fitz.Point(50, 1165), "El triangulo de la talla permite identificar de qué lado se encuentra la curva de la columna vertebral.\nEn su fotografia "+datos[8], fontsize=20)
        pagina.insert_image((165, 1230,1065, 1700),stream=imagenTriangulo, keep_proportion=True)
    else: 
        pagina.insert_text(fitz.Point(50, 1165), "El triangulo de la talla permite identificar de qué lado se encuentra la curva de la columna vertebral. \nEn su fotografia "+datos[8], fontsize=20)

    # Guardamos los cambios en el documento
    doc.write()
    # Guardamos el fichero PDF
    doc.save("nuevo.pdf", pretty=True)
    doc.close()
    return send_file("nuevo.pdf", mimetype='application/pdf')

if __name__== '__main__':
    scoliosisapp.run(host="0.0.0.0", port=4000)
