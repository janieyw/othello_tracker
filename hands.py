import cv2
import mediapipe as mp

class PlayerIdentification:
    def __init__(self):
        self.mp_hands = mp.solutions.hands
        self.hands = self.mp_hands.Hands()
        self.current_player = None

    def __del__(self):
        self.hands.close()

    def get_current_player(self, frame):
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
            # Iterate through each detected hand
            for hand_landmarks in results.multi_hand_landmarks:
                # Get the x coordinate of the wrist
                wrist_x = hand_landmarks.landmark[self.mp_hands.HandLandmark.WRIST].x * frame.shape[1]

                # Check if the wrist is on the left side of the frame
                if wrist_x < frame.shape[1] / 2:
                    self.current_player = 2
                else:
                    self.current_player = 1
        else:
            self.current_player = None

        return self.current_player

    def get_current_player_num(self):
        return self.current_player