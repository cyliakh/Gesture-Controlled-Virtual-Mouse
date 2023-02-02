import mediapipe as mp
import cv2
import mouse
import numpy as np
import math
import time

""" you can try this too : 
   def frame_pos2screen_pos(frame_size=(480, 640), screen_size=(768, 1366), frame_pos=None):
    x,y = screen_size[1]/frame_size[0], screen_size[0]/frame_size[1]    
    screen_pos = [frame_pos[0]*x, frame_pos[1]*y]
    return screen_pos
def euclidean(pt1, pt2):
    d = np.sqrt((pt1[0]-pt2[0])**2+(pt1[1]-pt2[1])**2)
    return d
euclidean((4, 3), (0, 0))

replace math.dist with euclidean and frame_pos2screen_pos down there for screen_pos"""

cam = cv2.VideoCapture(1)
fsize = (420, 620)
ssize = (2500, 3000) # change this according to your resolution and what works for you

#As the name, drawing_utils will draw landmark here and the hands will let us work with detection models.
mp_drawing = mp.solutions.drawing_utils
mp_hands = mp.solutions.hands
left, top, right, bottom = (200, 100, 500, 400)
events = ["sclick", "dclick", "rclick", "drag", "release"]

# count frames and check
check_every = 15
check_cnt = 0
last_event = None
out = cv2.VideoWriter("out.avi", cv2.VideoWriter_fourcc(*'XVID'), 30, (fsize[0], fsize[1]))
with mp_hands.Hands(static_image_mode=True,
                    max_num_hands=1,
                    min_detection_confidence=0.3) as hands:
    while cam.isOpened():
        ret, frame = cam.read()

        if not ret:
            continue

        #Flip the frame to look like selfie camera.
        frame = cv2.flip(frame, 1)
        #frame = cv2.resize(frame, (fsize[1], fsize[0]))

        h, w, _ = frame.shape

        #cv2.rectangle(frame, (left, top), (right, bottom), (0, 0, 255), 1)

        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        res = hands.process(rgb)

        #extracting landmarks of fingers. Like index finger’s tip, dip, middle and so on. There are overall 21 landmarks for each hand.
        if res.multi_hand_landmarks:
            for hand_landmarks in res.multi_hand_landmarks:
                index_tip = mp_drawing._normalized_to_pixel_coordinates(
                    hand_landmarks.landmark[mp_hands.HandLandmark.INDEX_FINGER_TIP].x,
                    hand_landmarks.landmark[mp_hands.HandLandmark.INDEX_FINGER_TIP].y,
                    w, h)

                index_dip = mp_drawing._normalized_to_pixel_coordinates(
                    hand_landmarks.landmark[mp_hands.HandLandmark.INDEX_FINGER_DIP].x,
                    hand_landmarks.landmark[mp_hands.HandLandmark.INDEX_FINGER_DIP].y,
                    w, h)

                index_pip = np.array(mp_drawing._normalized_to_pixel_coordinates(
                    hand_landmarks.landmark[mp_hands.HandLandmark.INDEX_FINGER_PIP].x,
                    hand_landmarks.landmark[mp_hands.HandLandmark.INDEX_FINGER_PIP].y,
                    w, h))

                thumb_tip = mp_drawing._normalized_to_pixel_coordinates(
                    hand_landmarks.landmark[mp_hands.HandLandmark.THUMB_TIP].x,
                    hand_landmarks.landmark[mp_hands.HandLandmark.THUMB_TIP].y,
                    w, h)

                middle_tip = mp_drawing._normalized_to_pixel_coordinates(
                    hand_landmarks.landmark[mp_hands.HandLandmark.MIDDLE_FINGER_TIP].x,
                    hand_landmarks.landmark[mp_hands.HandLandmark.MIDDLE_FINGER_TIP].y,
                    w, h)
                ring_tip = mp_drawing._normalized_to_pixel_coordinates(
                    hand_landmarks.landmark[mp_hands.HandLandmark.RING_FINGER_TIP].x,
                    hand_landmarks.landmark[mp_hands.HandLandmark.RING_FINGER_TIP].y,
                    w, h)

                """index_tipm = list(index_tip)
                index_tipm[0] = np.clip(index_tipm[0], left, right)
                index_tipm[1] = np.clip(index_tipm[1], top, bottom)
                
                index_tipm[0] = (index_tipm[0]-left) * fsize[0]/(right-left)
                index_tipm[1] = (index_tipm[1]-top) * fsize[1]/(bottom-top)"""

                if check_cnt == check_every:
                    if thumb_tip is not None and index_tip is not None and middle_tip is not None:
                        #If the distance between index finger’s tip and middle finger’s tip is less than 35 then double click
                        if math.dist(index_tip, middle_tip) < 40: #change the number, depending on your cam and resolution
                            last_event = "dclick"
                        else:
                            if last_event == "dclick":
                                last_event = None
                    if thumb_tip is not None and index_pip is not None:
                        if math.dist(thumb_tip, index_pip) < 60:
                            last_event = "sclick"
                        else:
                            if last_event == "sclick":
                                last_event = None
                    if thumb_tip is not None and index_tip is not None:
                        if math.dist(thumb_tip, index_tip) < 50:
                            last_event = "press"
                        else:
                            if last_event == "press":
                                last_event = "release"
                    if thumb_tip is not None and middle_tip is not None:
                        if math.dist(ring_tip, middle_tip) < 50:
                            last_event = "rclick"
                        else:
                            if last_event == "rclick":
                                last_event = None
                    check_cnt = 0
                    if check_cnt > 1:
                        last_event = None

                #screen_pos = np.interp(fsize, ssize, index_tipm)
                try:

                    screen_pos0 = ssize[0]*index_tip[0]/fsize[0]
                    screen_pos1 = ssize[1]*index_tip[1]/fsize[1]
                except TypeError:
                    print("Oops! Not valid lol ")


                #print(screen_pos0, screen_pos1)

                pos0 = int(screen_pos0)
                pos1 = int(screen_pos1)
                mouse.move(int(pos0), int(pos1))
                time.sleep(0.05)

                if check_cnt == 0:
                    if last_event == "sclick":
                        mouse.click()
                    elif last_event == "rclick":
                        mouse.right_click()
                    elif last_event == "dclick":
                        mouse.double_click()
                    elif last_event == "press":
                        mouse.press()
                    else:
                        mouse.release()
                    print(last_event)
                    cv2.putText(frame, last_event, (20, 20),
                                cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
                check_cnt += 1
                mp_drawing.draw_landmarks(
                    frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)

        cv2.imshow("Window", frame)
        out.write(frame)

        key = cv2.waitKey(3)
        if key == ord('q'):
            break
    cv2.destroyAllWindows()

cam.release()
out.release()
cv2.destroyAllWindows()
