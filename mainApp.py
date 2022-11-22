import time

import numpy as np
import cv2
from keras.models import load_model
from pyfirmata import Arduino,util,  SERVO

#ativa o arduino
board = Arduino("COM3")
it = util.Iterator(board)
it.start()

#porta onde esta o servo
pin = 9

#configurando o servo motor
board.digital[pin].mode = SERVO

#configurando o motor dc(move a esteira)
motor = board.get_pin("d:7:o")

#para ativar a camera
cap = cv2.VideoCapture(0)

#carrega o modelo treinado pela IA
model = load_model('modelo.h5')

#variaveis para validar a seleção para o braco da esteira
validacao = 0
cont = 0

#converte as fotos em tons de cinza
def escalaDeCinza(img):
    img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    return img

#trata as fotos
def equalizar(img):
    img = cv2.equalizeHist(img)
    return img

#pre-processa as fotos
def preProcessamento(img):
    img = escalaDeCinza(img)
    img = equalizar(img)
    img = img / 255
    return img

#retorna o nome da classe definida pela IA
def getNomeClasse(classNo):
    if classNo == 0:
        return 'Neutro'
    elif classNo == 1:
        return 'NaoFertil'
    elif classNo == 2:
        return 'Fertil'

#move ou para a esteira
def moverEsteira(escolha):
    if escolha == "mover":
        motor.write(1)
    elif escolha == "parar":
        motor.write(0)

#rotaciona o servo motor
def rotacionarBraco(pin, angle):
    board.digital[pin].write(angle)

#inicia o programa colocando o braço em uma posicao padrao
for i in range(0, 80):
    rotacionarBraco(pin, i)

#move a esteira
moverEsteira("mover")

while True:

    #liga a camera
    success, imgOrignal = cap.read()

    #redimenciona as imagens
    img = np.asarray(imgOrignal)
    img = cv2.resize(img, (32, 32))
    img = preProcessamento(img)
    img = img.reshape(1, 32, 32, 1)

    #preve usando o modelo pre-treinado
    predictions = model.predict(img)

    #adiciona em variaveis o numero da classe prevista e a probabilidade
    numeroClasse = np.argmax(predictions)
    valorProbabilidade = np.amax(predictions)

    #caso a validacao for igual o numero da classe e a probabilidade for superior a 50% o contador é adicionado
    print("numero do contador:", cont)
    if numeroClasse == validacao and valorProbabilidade >= 0.50:
        cont += 1

    #caso a validacao for diferente ou tiver baixa probabilidade, o contador é resetado
    if numeroClasse != validacao or valorProbabilidade < 0.50:
        cont = 0

    #evitar que a esteira pare no neutro
    if numeroClasse == 0 and cont == 10:
        moverEsteira("mover")

    #mostra na tela do console o numero da classe e a probabilidade
    print(numeroClasse, valorProbabilidade)

    #caso a classe for diferente de zero(a neutra) ele vai para a esteira para a leitura do ovo
    if numeroClasse != 0 and cont == 3:
        moverEsteira("parar")

    #caso o ovo tiver uma validação de 30 vezes, ele vai mover o braço, mover a esteira e para a leitura por 3 segs
    if numeroClasse == 1 and cont == 30:
        for i in range(0, 80):
            rotacionarBraco(pin, i)
        moverEsteira("mover")
        time.sleep(3)
    elif numeroClasse == 2 and cont == 30:
        for i in range(80, 0, -1):
            rotacionarBraco(pin, i)
        moverEsteira("mover")
        time.sleep(3)

    #coloca na tela da camera as informações das predicoes
    cv2.putText(imgOrignal, str(getNomeClasse(numeroClasse)), (120, 70), cv2.FONT_HERSHEY_SIMPLEX, 2,
                (0, 0, 255), 8, cv2.LINE_AA)
    cv2.putText(imgOrignal, str(round(valorProbabilidade * 100, 2)) + "%", (120, 110), cv2.FONT_HERSHEY_SIMPLEX, 0.75, (0, 0, 255), 2,
                cv2.LINE_AA)

    cv2.imshow("Result", imgOrignal)

    # validacao vai recever o valor da classe para modificar o contador
    validacao = numeroClasse

    #ao apertar ESC a esteira para e o programa é encerrado
    if cv2.waitKey(1) == 27:
        moverEsteira("parar")
        cv2.destroyAllWindows()
        break
