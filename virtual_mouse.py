import cv2
import numpy as np
import mediapipe as mp   # type: ignore

import pyautogui   # type: ignore

import time
import pyttsx3   # type: ignore

import keyboard  # <-- ADDED for reliable 'q' key detection   # type: ignore


# Initialize pyttsx3 for voice feedback
engine = pyttsx3.init()

# Initialize mediapipe hand detection
mp_drawing = mp.solutions.drawing_utils
mp_hands = mp.solutions.hands
hands = mp_hands.Hands(max_num_hands=1)
screen_width, screen_height = pyautogui.size()
prev_time = 0

# Initialize VideoCapture and output writer for screen recording
cap = cv2.VideoCapture(0)
fourcc = cv2.VideoWriter_fourcc(*"XVID")
out = cv2.VideoWriter("screen_record.avi", fourcc, 8.0, (screen_width, screen_height))

while cap.isOpened():
    success, image = cap.read()
    if not success:
        continue

    image = cv2.flip(image, 1)
    frame_height, frame_width, _ = image.shape
    rgb_frame = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    result = hands.process(rgb_frame)

    if result.multi_hand_landmarks:
        for hand_landmarks in result.multi_hand_landmarks:
            landmarks = hand_landmarks.landmark

            index_finger = landmarks[8]
            thumb = landmarks[4]
            middle_finger = landmarks[12]

            x = int(index_finger.x * frame_width)
            y = int(index_finger.y * frame_height)

            screen_x = int(screen_width * index_finger.x)
            screen_y = int(screen_height * index_finger.y)

            pyautogui.moveTo(screen_x, screen_y)

            # Draw visual cursor
            cv2.circle(image, (x, y), 15, (255, 0, 0), cv2.FILLED)

            index_y = int(index_finger.y * frame_height)
            thumb_y = int(thumb.y * frame_height)
            middle_y = int(middle_finger.y * frame_height)

            # Left click
            if abs(index_y - thumb_y) < 30:
                pyautogui.click()
                engine.say("Clicked")
                engine.runAndWait()
                pyautogui.sleep(1)

            # Right click
            if abs(middle_y - thumb_y) < 30:
                pyautogui.rightClick()
                engine.say("Right Clicked")
                engine.runAndWait()
                pyautogui.sleep(1)

            # Scrolling
            finger_y_diff = abs(landmarks[8].y - landmarks[12].y) * frame_height
            if finger_y_diff < 20:
                pyautogui.scroll(20)
                engine.say("Scrolling Up")
                engine.runAndWait()
            elif finger_y_diff > 80:
                pyautogui.scroll(-20)
                engine.say("Scrolling Down")
                engine.runAndWait()

            mp_drawing.draw_landmarks(image, hand_landmarks, mp_hands.HAND_CONNECTIONS)

    # FPS counter
    curr_time = time.time()
    fps = 1 / (curr_time - prev_time)
    prev_time = curr_time
    cv2.putText(image, f'FPS: {int(fps)}', (20, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0,255,0), 2)

    # Display
    cv2.imshow("Virtual Mouse", image)

    # Optional: record screen activity
    screenshot = pyautogui.screenshot()
    frame_np = np.array(screenshot)
    frame_bgr = cv2.cvtColor(frame_np, cv2.COLOR_RGB2BGR)
    out.write(frame_bgr)

    # Exit when 'q' is pressed (more reliable using keyboard lib)
    if cv2.waitKey(1) & 0xFF == ord('q') or keyboard.is_pressed('q'):
        engine.say("Exiting Virtual Mouse")
        engine.runAndWait()
        break

cap.release()
out.release()
cv2.destroyAllWindows()
