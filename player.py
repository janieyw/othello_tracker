import cv2
import mediapipe as mp
import numpy as np

class Player:
    def __init__(self):
        self.mp_hands = mp.solutions.hands
        self.hands = self.mp_hands.Hands()
        self.current_player = None

    def __del__(self):
        self.hands.close()

    def get_current_player_num(self, frame):
        # Flip the frame horizontally
        frame = cv2.flip(frame, 1)

        # Convert the color from BGR to RGB
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        # To improve performance, optionally mark the image as not writeable to pass by reference.
        frame.flags.writeable = False

        # Process the frame with the Mediapipe hand detection model
        results = self.hands.process(frame)

        # Set the frame to writeable again
        frame.flags.writeable = True

        if results.multi_hand_landmarks:
            # Initialize a list of hand coordinates
            hand_coords = []

            # Iterate through each detected hand
            for hand_landmarks in results.multi_hand_landmarks:
                # Get the x and y coordinates of the wrist
                wrist_x = hand_landmarks.landmark[self.mp_hands.HandLandmark.WRIST].x * frame.shape[1]
                wrist_y = hand_landmarks.landmark[self.mp_hands.HandLandmark.WRIST].y * frame.shape[0]

                # Append the hand coordinates to the list
                hand_coords.append((wrist_x, wrist_y))

            # Determine which hand is closest to the center of the frame
            center_x = frame.shape[1] / 2
            center_y = frame.shape[0] / 2
            distances = [np.sqrt((x - center_x)**2 + (y - center_y)**2) for x, y in hand_coords]
            closest_hand_index = np.argmin(distances)

            # Determine the player number based on the closest hand
            closest_hand_x, closest_hand_y = hand_coords[closest_hand_index]
            if closest_hand_x < center_x:
                self.current_player = 2
            else:
                self.current_player = 1
        else:
            self.current_player = None

        return self.current_player

    def get_right_player_num(self, prev_player_num):
        if prev_player_num == None:
            right_player_num = 1
        elif prev_player_num == 1:
            right_player_num = 2
        else:  # prev_player_num == 2
            right_player_num = 1
        return right_player_num
