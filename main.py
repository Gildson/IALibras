import numpy as np
import mediapipe as mp
from mediapipe.tasks.python import vision
import cv2
from mediapipe.framework.formats import landmark_pb2


def detecta_letra(frame: np.array, pontos:dict) -> None:
    """
    Função para detectar a letra na linguagem de sinais (Libras).

    Essa função detecta, utilizando os pontos das mãos, qual
    letra foi feita. Utilizando a localização x e y do ponto,
    detectamos a letra feita.

    Attributes:
        frame (np.array): Array com os pixels de cada frame do vídeo.

    Return:
        None.
    """

    try:
        # ------------------------------ letra A ------------------------------------
        if (
            (pontos.get(6)[1] < pontos.get(7)[1] and pontos.get(7)[1] < pontos.get(8)[1]) and 
            (pontos.get(10)[1] < pontos.get(11)[1] and pontos.get(11)[1] < pontos.get(12)[1]) and 
            (pontos.get(14)[1] < pontos.get(15)[1] and pontos.get(15)[1] < pontos.get(16)[1]) and 
            (pontos.get(18)[1] < pontos.get(19)[1] and pontos.get(19)[1] < pontos.get(20)[1]) and 
            (pontos.get(4)[1] < pontos.get(6)[1]) and 
            (pontos.get(3)[1] < pontos.get(7)[1])
        ):
            cv2.putText(frame, "Letra A", (int(largura/2)-100, 40), cv2.FONT_HERSHEY_DUPLEX, 0.7, (255,255,255), 2)
        # ------------------------------ letra B ------------------------------------
        if (
            (pontos.get(6)[1] > pontos.get(7)[1] and pontos.get(7)[1] > pontos.get(8)[1]) and 
            (pontos.get(10)[1] > pontos.get(11)[1] and pontos.get(11)[1] > pontos.get(12)[1]) and 
            (pontos.get(14)[1] > pontos.get(15)[1] and pontos.get(15)[1] > pontos.get(16)[1]) and 
            (pontos.get(18)[1] > pontos.get(19)[1] and pontos.get(19)[1] > pontos.get(20)[1]) and 
            (pontos.get(3)[0] < pontos.get(2)[0]) and (pontos.get(4)[0] < pontos.get(5)[0])
        ):
            cv2.putText(frame, "Letra B", (int(largura/2)-100, 40), cv2.FONT_HERSHEY_DUPLEX, 0.7, (255,255,255), 2)
    except Exception as e:
        pass
        
model_path = r'C:\Users\Gildson\Documents\GestureRecognizier\hand_landmarker.task'

# Variáveis para configuração do mediapipe
BaseOptions = mp.tasks.BaseOptions
HandLandmarker = vision.HandLandmarker
HandLandmarkerOptions = vision.HandLandmarkerOptions
VisionRunningMode = vision.RunningMode

options = HandLandmarkerOptions(
    base_options=BaseOptions(model_asset_path=model_path),
    running_mode=VisionRunningMode.VIDEO,
    num_hands=2)

landmarker = HandLandmarker.create_from_options(options)

# Captura da webcam
cap = cv2.VideoCapture(0)

while True:
    ret, frame = cap.read()

    comprimento, largura, _ = frame.shape

    if not ret:
        break
    
    frame_timestamp_ms = int(cap.get(cv2.CAP_PROP_POS_MSEC))
    
    mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=frame)

    hand_landmarker_result = landmarker.detect_for_video(mp_image, frame_timestamp_ms)

    hand_landmarks_list = hand_landmarker_result.hand_landmarks
    for idx in range(len(hand_landmarks_list)):
        hand_landmarks = hand_landmarks_list[idx]
        
        # Desenha os pontos da mão.
        hand_landmarks_proto = landmark_pb2.NormalizedLandmarkList()
        hand_landmarks_proto.landmark.extend(
            [landmark_pb2.NormalizedLandmark(x=landmark.x, y=landmark.y, z=landmark.z) for landmark in hand_landmarks]
            )
        hand = hand_landmarks_proto.landmark
        mp.solutions.drawing_utils.draw_landmarks(
            frame,
            hand_landmarks_proto,
            landmark_drawing_spec=mp.solutions.drawing_utils.DrawingSpec(
                    color=(255,255,0),
                    thickness=1,
                    circle_radius=2
                ),
                connection_drawing_spec=mp.solutions.drawing_utils.DrawingSpec(
                    color=(0,0,255),
                    thickness=1,
                    circle_radius=2
                )
            )
        pontos = {}
        # Divide a mão em seis partes
        for id_coord, coord_xyz in enumerate(hand):
            coord_cv = mp.solutions.drawing_utils._normalized_to_pixel_coordinates(
                coord_xyz.x,
                coord_xyz.y,
                largura,
                comprimento
            )
            try:
                # ---------------------------------- Dedo polegar ---------------------------------------------
                if id_coord == 2:
                    x_2, y_2 = coord_cv
                    pontos.update({2:[x_2,y_2]})
                if id_coord == 3:
                    x_3, y_3 = coord_cv
                    pontos.update({3:[x_3,y_3]})
                if id_coord == 4:
                    x_4, y_4 = coord_cv
                    pontos.update({4:[x_4,y_4]})
                # ----------------------------------  Dedo indicador  ---------------------------------------
                if id_coord == 6:
                    x_6, y_6 = coord_cv
                    pontos.update({6:[x_6,y_6]})
                if id_coord == 7:
                    x_7, y_7 = coord_cv
                    pontos.update({7:[x_7,y_7]})
                if id_coord == 8:
                    x_8, y_8 = coord_cv
                    pontos.update({8:[x_8,y_8]})
                # ---------------------------------- Dedo medio ---------------------------------------------
                if id_coord == 10:
                    x_10, y_10 = coord_cv
                    pontos.update({10:[x_10,y_10]})
                if id_coord == 11:
                    x_11, y_11 = coord_cv
                    pontos.update({11:[x_11,y_11]})
                if id_coord == 12:
                    x_12, y_12 = coord_cv
                    pontos.update({12:[x_12,y_12]})
                # ---------------------------------- Dedo anelar ---------------------------------------------
                if id_coord == 14:
                    x_14, y_14 = coord_cv
                    pontos.update({14:[x_14,y_14]})
                if id_coord == 15:
                    x_15, y_15 = coord_cv
                    pontos.update({15:[x_15,y_15]})
                if id_coord == 16:
                    x_16, y_16 = coord_cv
                    pontos.update({16:[x_16,y_16]})
                # ---------------------------------- Dedo minimo ---------------------------------------------
                if id_coord == 18:
                    x_18, y_18 = coord_cv
                    pontos.update({18:[x_18,y_18]})
                if id_coord == 19:
                    x_19, y_19 = coord_cv
                    pontos.update({19:[x_19,y_19]})
                if id_coord == 20:
                    x_20, y_20 = coord_cv
                    pontos.update({20:[x_20,y_20]})
                # ---------------------------------- Palma da mão ---------------------------------------------
                if id_coord == 0:
                    x_0, y_0 = coord_cv
                    pontos.update({0:[x_0,y_0]})
                if id_coord == 1:
                    x_1, y_1 = coord_cv
                    pontos.update({1:[x_1,y_1]})
                if id_coord == 5:
                    x_5, y_5 = coord_cv
                    pontos.update({5:[x_5,y_5]})
                if id_coord == 9:
                    x_9, y_9 = coord_cv
                    pontos.update({9:[x_9,y_9]})
                if id_coord == 13:
                    x_13, y_13 = coord_cv
                    pontos.update({13:[x_13,y_13]})
                if id_coord == 17:
                    x_17, y_17 = coord_cv
                    pontos.update({17:[x_17,y_17]})
            except:
                pass                                                
            # if id_coord == 3 or id_coord == 2:
            #     cv2.putText(frame, f"{coord_cv[0]}", (coord_cv[0], coord_cv[1]), cv2.FONT_HERSHEY_DUPLEX, 0.9, (0,0,0), 2)
        detecta_letra(frame,pontos)
    # detecta qual mão está sendo mostrada no vídeo
    try:
        if hand_landmarker_result.handedness[0][0].index == 1:
            cv2.putText(frame, f"Hand: {hand_landmarker_result.handedness[0][0].category_name}", (1, 20), cv2.FONT_HERSHEY_DUPLEX, 0.9, (255,255,255), 2)
        elif hand_landmarker_result.handedness[0][0].index == 0:
            cv2.putText(frame, f"Hand: {hand_landmarker_result.handedness[0][0].category_name}", (largura-180, 20), cv2.FONT_HERSHEY_DUPLEX, 0.9, (255,255,255), 2)
    except Exception as e:
        cv2.putText(frame, "Not hand", (int(largura/2)-50, 20), cv2.FONT_HERSHEY_DUPLEX, 0.9, (255,255,255), 2)
        pass

    # Mostra o vídeo
    cv2.imshow('video', frame)
    # Para finalizar o vídeo apertando a tecla 'c'
    if cv2.waitKey(10) & 0xFF == ord('c'):
        break
        
# Libera os recursos utilizados pelo OpenCV    
cap.release()
cv2.destroyAllWindows()