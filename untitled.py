import os
import cv2
import mediapipe as mp
import pandas as pd
import numpy as np

# configuration
IMAGE_DIR = 'C:\Users\prawb\OneDrive\Documents\ASL_Training_images'
OUTPUT_CSV = 'asl_landmarks.csv'

# initialize mediapipe
mp_hands = mp.solutions.hands
hands = mp_hands.Hands(
    static_image_mode = True,
    max_num_hands = 1,
    min_detection_confidence = 0.7)

def normalize_landmarks(landmark_list):
    """
    Translates landmarks so the wrist (point 0) is at (0,0) and scales them to a consisten range.
    """
    temp_landmark_list =[]

    # convert to relative coordinates based on writs (point 0)
    base_x, base_y = landmark_list[0][0], landmark_list[0][1]

    for x,y in landmark_list:
        temp_landmark_list.append([x - base_x, y - base_y])

    # flatten the list [x0,y0,x1,y1....]
    flat_list = [item for sublist in temp_landmark_list for item in sublist]

    # normalize by the maximum absolute value found in the list
    max_val = max(max(abs(x) for x in flat_list), 1e-6) #avoid division by zero
    return [val/max_val for val in flat_list]

def process_images():
    data_list =[]

    #  get sorted list of labels (folder names)
    labels = sorted ([l for l in os.listdir(IMAGE_DIR) if os.path.isdir(os.path.join(IMAGE_DIR, l))])

    for label in labels:
        label_path = os.path.join(IMAGE_DIR, label)
        print(f"Processing letter: {label}...")

        for img_name in os.listdir(label_path):
            img_path = os.path.join(label_path, img_name)
            img = cv2.imread(img_path)
            if img is None:
                continue

            # mediapipe processing
            img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            result = hands.process(img_rgb)

            if result.multi_hand_landmarks:
                for hand_landmarks in result.multi_hand_landmarks:
                    # extract raw(x,y)
                    coords = [[lm.x, lm.y] for lm in hand_landmarks.landmark]

                    # apply normalization
                    normalized_coords = normalize_landmarks(coords)

                    # store result: label + 42 features
                    data_list.append([label] + normalized_coords)

    # define columns
    columns = ['label']
    for i in range(21):
        columns.extend([f'x{i}',f'y{i}'])

    # save to csv
    df = pd.DataFrame(data_list,columns = columns)
    df.to_csv(OUTPUT_CSV, index = False)
    print(f"\nSuccess! Created {OUTPUT_CSV} with {len(df)} samples.")

if __name__ == "__main__":
    process_images()
