import cv2
from tkinter import Tk, Label, Button, Entry, StringVar
from deepface import DeepFace
import os
import face_recognition
import pyttsx3
import numpy as np
from sklearn.cluster import KMeans

# Global Variables
user_name = ""
images_captured = 0
dataset_path = "user_datasets"
known_face_encodings = []
known_face_names = []

# Initialize text-to-speech and dataset directory
engine = pyttsx3.init()
if not os.path.exists(dataset_path):
    os.makedirs(dataset_path)

def get_dominant_color(image, face_location):
    """Get the dominant color of clothes below the face."""
    top, right, bottom, left = face_location
    face_height = bottom - top
    face_width = right - left
    
    # Define the region of interest (clothes area below the face)
    # Start further down from the face to avoid neck/skin areas
    clothes_top = bottom + int(face_height * 0.2)
    # Limit the bottom to avoid going too far
    clothes_bottom = min(clothes_top + int(face_height * 1.5), image.shape[0])
    # Extend width to capture more of the clothing
    clothes_left = max(0, left - int(face_width * 0.3))
    clothes_right = min(right + int(face_width * 0.3), image.shape[1])
    
    # Check if region is valid
    if clothes_bottom <= clothes_top or clothes_right <= clothes_left:
        return None
    
    # Extract the clothes region
    clothes_region = image[clothes_top:clothes_bottom, clothes_left:clothes_right]
    
    # If region is too small, return None
    if clothes_region.shape[0] < 10 or clothes_region.shape[1] < 10:
        return None
    
    # Create a mask to filter out skin tones
    hsv_clothes = cv2.cvtColor(clothes_region, cv2.COLOR_BGR2HSV)
    # Skin tone range in HSV (broad range)
    lower_skin = np.array([0, 20, 70], dtype=np.uint8)
    upper_skin = np.array([30, 150, 255], dtype=np.uint8)
    skin_mask = cv2.inRange(hsv_clothes, lower_skin, upper_skin)
    # Invert mask to keep non-skin pixels
    clothes_mask = cv2.bitwise_not(skin_mask)
    
    # Apply mask to clothes region
    masked_clothes = cv2.bitwise_and(clothes_region, clothes_region, mask=clothes_mask)
    
    # Reshape for clustering, filter out black pixels (masked areas)
    pixels = masked_clothes.reshape((-1, 3))
    pixels = pixels[~np.all(pixels == [0, 0, 0], axis=1)]
    
    # If no valid pixels remain, return None
    if len(pixels) == 0:
        return None
    
    # Use KMeans to find dominant colors
    kmeans = KMeans(n_clusters=3, n_init=10)
    kmeans.fit(pixels)
    
    # Get the colors and their counts
    colors = kmeans.cluster_centers_
    labels = kmeans.labels_
    counts = np.bincount(labels)
    
    # Get top 3 colors
    color_frequencies = [(count, color) for count, color in zip(counts, colors)]
    color_frequencies.sort(reverse=True)
    
    # Return the most common non-skin color
    for count, color in color_frequencies:
        color_name = get_color_name(color)
        if color_name not in ["skin", "light skin", "tan"]:
            return color_name
    
    # If all detected colors are skin-like, return the most dominant
    return get_color_name(color_frequencies[0][1])

def get_color_name(rgb):
    """Convert RGB values to color name."""
    b, g, r = rgb
    
    # Define color ranges with improved boundaries
    colors = {
        "red": (np.array([0, 0, 150]), np.array([80, 80, 255])),
        "burgundy": (np.array([20, 20, 100]), np.array([100, 60, 180])),
        "green": (np.array([60, 150, 60]), np.array([180, 255, 180])),
        "dark green": (np.array([20, 70, 20]), np.array([100, 150, 100])),
        "blue": (np.array([150, 40, 40]), np.array([255, 120, 120])),
        "navy blue": (np.array([80, 20, 20]), np.array([150, 70, 70])),
        "light blue": (np.array([200, 170, 100]), np.array([255, 220, 170])),
        "yellow": (np.array([0, 180, 180]), np.array([100, 255, 255])),
        "purple": (np.array([120, 40, 120]), np.array([200, 100, 200])),
        "lavender": (np.array([180, 160, 200]), np.array([230, 200, 255])),
        "orange": (np.array([0, 80, 180]), np.array([100, 170, 255])),
        "pink": (np.array([160, 120, 220]), np.array([255, 200, 255])),
        "black": (np.array([0, 0, 0]), np.array([40, 40, 40])),
        "white": (np.array([200, 200, 200]), np.array([255, 255, 255])),
        "gray": (np.array([80, 80, 80]), np.array([180, 180, 180])),
        "light gray": (np.array([180, 180, 180]), np.array([220, 220, 220])),
        "brown": (np.array([40, 60, 80]), np.array([120, 140, 160])),
        "tan": (np.array([140, 160, 180]), np.array([200, 220, 240])),
        "cream": (np.array([200, 200, 180]), np.array([255, 255, 230])),
        "skin": (np.array([80, 120, 160]), np.array([140, 180, 220])),
        "light skin": (np.array([160, 170, 200]), np.array([220, 230, 255])),
    }
    
    # Check which color range the RGB values fall into
    for color_name, (lower, upper) in colors.items():
        if np.all(rgb >= lower) and np.all(rgb <= upper):
            return color_name
    
    # Calculate color properties
    brightness = np.mean(rgb)
    max_channel = np.argmax(rgb)
    
    # Determine base color based on dominant channel
    if max_channel == 0:  # B is max
        base_color = "blue"
    elif max_channel == 1:  # G is max
        if rgb[1] > rgb[0] + rgb[2]:
            base_color = "green"
        else:
            base_color = "teal"
    else:  # R is max
        if rgb[2] > rgb[0] + rgb[1]:
            base_color = "red"
        elif rgb[1] > rgb[0]:
            base_color = "orange"
        else:
            base_color = "pink"
    
    # Add brightness modifier
    if brightness < 70:
        return f"dark {base_color}"
    elif brightness > 180:
        return f"light {base_color}"
    else:
        return base_color

def load_known_faces():
    """Load all known face encodings from the dataset."""
    global known_face_encodings, known_face_names
    known_face_encodings = []
    known_face_names = []
    
    for user in os.listdir(dataset_path):
        user_folder = os.path.join(dataset_path, user)
        if not os.path.isdir(user_folder):
            continue
            
        for image_name in os.listdir(user_folder):
            image_path = os.path.join(user_folder, image_name)
            try:
                image = face_recognition.load_image_file(image_path)
                face_locations = face_recognition.face_locations(image)
                
                if face_locations:
                    encoding = face_recognition.face_encodings(image, [face_locations[0]])[0]
                    known_face_encodings.append(encoding)
                    known_face_names.append(user)
                    print(f"Loaded face data: {user} - {image_name}")
            except Exception as e:
                print(f"Error loading {image_path}: {str(e)}")

def home_page():
    """Create the home page."""
    home = Tk()
    home.title("Face Recognizer")
    home.geometry("400x300")
    Label(home, text="Face Recognizer", font=("Arial", 20)).pack(pady=20)
    Button(home, text="Add a User", width=20, command=lambda: add_user_page(home)).pack(pady=10)
    Button(home, text="Check a User", width=20, command=lambda: check_user_page(home)).pack(pady=10)
    home.mainloop()

def add_user_page(home):
    """Create the add user page."""
    home.destroy()
    add_user = Tk()
    add_user.title("Add a User")
    add_user.geometry("400x300")
    Label(add_user, text="Enter User Name:", font=("Arial", 14)).pack(pady=10)
    name_var = StringVar()
    Entry(add_user, textvariable=name_var, font=("Arial", 14)).pack(pady=10)
    Button(add_user, text="Next", width=20, command=lambda: capture_dataset_page(add_user, name_var.get())).pack(pady=10)
    Button(add_user, text="Back", width=20, command=lambda: (add_user.destroy(), home_page())).pack(pady=10)
    add_user.mainloop()

def capture_dataset_page(add_user, name):
    """Create the capture dataset page."""
    global user_name, images_captured
    add_user.destroy()
    user_name = name.strip()
    images_captured = 0
    
    capture_page = Tk()
    capture_page.title("Capture Dataset")
    capture_page.geometry("400x300")
    Label(capture_page, text=f"User: {user_name}", font=("Arial", 14)).pack(pady=10)
    count_label = Label(capture_page, text=f"Images captured: {images_captured}", font=("Arial", 14))
    count_label.pack(pady=10)
    Button(capture_page, text="Capture Dataset", width=20, command=lambda: capture_images(count_label)).pack(pady=10)
    Button(capture_page, text="Train Model", width=20, command=lambda: train_model()).pack(pady=10)
    Button(capture_page, text="Home", width=20, command=lambda: (capture_page.destroy(), home_page())).pack(pady=10)
    capture_page.mainloop()

def capture_images(count_label):
    """Capture images of the user."""
    global images_captured, user_name
    if not user_name:
        print("Error: User name is empty")
        return

    user_folder = os.path.join(dataset_path, user_name)
    if not os.path.exists(user_folder):
        os.makedirs(user_folder)

    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("Error: Cannot open camera")
        return

    face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

    while images_captured < 20:  # Reduced for faster processing
        ret, frame = cap.read()
        if not ret:
            break

        gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = face_cascade.detectMultiScale(gray_frame, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))

        for (x, y, w, h) in faces:
            face_roi = frame[y:y+h, x:x+w]
            
            # Ensure minimum face size
            if w < 100 or h < 100:
                continue
                
            image_path = os.path.join(user_folder, f"{images_captured}.jpg")
            cv2.imwrite(image_path, face_roi)

            try:
                result = DeepFace.analyze(face_roi, actions=['emotion'], enforce_detection=False)
                emotion = result[0]['dominant_emotion']
                
                engine.say(f"{user_name} is {emotion}")
                engine.runAndWait()

                cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 2)
                cv2.putText(frame, f"{emotion}", (x, y-10), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
                
                images_captured += 1
                count_label.config(text=f"Images captured: {images_captured}")
            except Exception as e:
                print(f"Error analyzing face: {str(e)}")

        cv2.imshow("Capture Images", frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

def train_model():
    """Train the face recognition model."""
    load_known_faces()
    print("Model trained successfully!")
    engine.say("Training completed")
    engine.runAndWait()

def check_user_page(home):
    """Create the check user page."""
    home.destroy()
    check_user = Tk()
    check_user.title("Check a User")
    check_user.geometry("400x300")
    Label(check_user, text="Check User", font=("Arial", 14)).pack(pady=10)
    Button(check_user, text="Recognize User", width=20, command=recognize_user).pack(pady=10)
    Button(check_user, text="Home", width=20, command=lambda: (check_user.destroy(), home_page())).pack(pady=10)
    check_user.mainloop()

def recognize_user():
    """Recognize the user from the webcam."""
    if not known_face_encodings:
        load_known_faces()
        if not known_face_encodings:
            print("No face data available")
            engine.say("No users found in database")
            engine.runAndWait()
            return

    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("Error: Cannot open camera")
        return

    last_spoken_info = None
    speech_cooldown = 0
    frame_count = 0
    color_stabilization = {}  # Dictionary to track recent colors
    
    while True:
        ret, frame = cap.read()
        if not ret:
            break

        frame_count += 1
        if frame_count % 3 != 0:  # Process every 3rd frame
            continue

        face_locations = face_recognition.face_locations(frame)
        if face_locations:
            face_encodings = face_recognition.face_encodings(frame, face_locations)

            for face_location, face_encoding in zip(face_locations, face_encodings):
                matches = face_recognition.compare_faces(known_face_encodings, face_encoding, tolerance=0.5)
                face_distances = face_recognition.face_distance(known_face_encodings, face_encoding)
                
                name = "Unknown"
                if len(face_distances) > 0:
                    best_match_index = np.argmin(face_distances)
                    if matches[best_match_index] and face_distances[best_match_index] < 0.5:
                        name = known_face_names[best_match_index]
                        
                        # Get clothing color
                        clothing_color = get_dominant_color(frame, face_location)
                        
                        # Stabilize color detection to prevent flickering
                        if name not in color_stabilization:
                            color_stabilization[name] = {"colors": [], "frames_since_update": 0}
                            
                        if clothing_color:
                            color_stabilization[name]["colors"].append(clothing_color)
                            color_stabilization[name]["colors"] = color_stabilization[name]["colors"][-5:]  # Keep last 5 colors
                            
                        # Get the most common color from recent frames
                        if color_stabilization[name]["colors"]:
                            color_counts = {}
                            for color in color_stabilization[name]["colors"]:
                                if color in color_counts:
                                    color_counts[color] += 1
                                else:
                                    color_counts[color] = 1
                            stable_color = max(color_counts, key=color_counts.get)
                        else:
                            stable_color = None
                        
                        color_text = f" wearing {stable_color}" if stable_color else ""
                        current_info = f"{name}{color_text}"
                        
                        # Update speech cooldown
                        color_stabilization[name]["frames_since_update"] += 1
                        if current_info != last_spoken_info and speech_cooldown <= 0 and color_stabilization[name]["frames_since_update"] > 15:
                            greeting = f"Hello {name}"
                            if stable_color:
                                greeting += f", I see you're wearing {stable_color} today"
                            
                            engine.say(greeting)
                            engine.runAndWait()
                            last_spoken_info = current_info
                            speech_cooldown = 30
                            color_stabilization[name]["frames_since_update"] = 0

                top, right, bottom, left = face_location
                cv2.rectangle(frame, (left, top), (right, bottom), (0, 255, 0), 2)
                
                # Show name
                cv2.putText(frame, name, (left, top-10), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
                
                # Calculate clothing area
                face_height = bottom - top
                face_width = right - left
                clothes_top = bottom + int(face_height * 0.2)
                clothes_bottom = min(clothes_top + int(face_height * 1.5), frame.shape[0])
                clothes_left = max(0, left - int(face_width * 0.3))
                clothes_right = min(right + int(face_width * 0.3), frame.shape[1])
                
                # Get stable color for display
                if name in color_stabilization and color_stabilization[name]["colors"]:
                    color_counts = {}
                    for color in color_stabilization[name]["colors"]:
                        if color in color_counts:
                            color_counts[color] += 1
                        else:
                            color_counts[color] = 1
                    display_color = max(color_counts, key=color_counts.get)
                else:
                    display_color = "analyzing..."
                
                # Draw rectangle over detected clothing area
                cv2.rectangle(frame, (clothes_left, clothes_top), 
                             (clothes_right, clothes_bottom), (255, 0, 0), 2)
                
                # Show color info
                cv2.putText(frame, f"Clothing: {display_color}", 
                           (clothes_left, clothes_bottom + 20), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 0, 0), 2)

        speech_cooldown = max(0, speech_cooldown - 1)
        cv2.imshow("Face Recognition", frame)
        
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    home_page()