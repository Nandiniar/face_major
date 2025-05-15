import cv2
import os
import numpy as np
import face_recognition
import pyttsx3
import datetime
import json
import time
from threading import Thread
from collections import deque

# Global Variables
user_name = ""
images_captured = 0
dataset_path = "user_datasets"
visits_data_path = "visits_data"
known_face_encodings = []
known_face_names = []

# Initialize text-to-speech engine
engine = pyttsx3.init()
# Lower rate for better clarity
engine.setProperty('rate', 150)

# Create necessary directories
if not os.path.exists(dataset_path):
    os.makedirs(dataset_path)
if not os.path.exists(visits_data_path):
    os.makedirs(visits_data_path)

class AsyncFaceAnalyzer:
    """Class to handle face analysis in a separate thread"""
    def __init__(self):
        self.is_analyzing = False
        self.queue = []
        self.results = {}
        
    def analyze_emotion(self, face_img, user_id):
        """Queue a face for emotion analysis"""
        if not self.is_analyzing:
            self.queue.append((face_img, user_id))
            
    def start_worker(self):
        """Start the worker thread"""
        Thread(target=self._worker, daemon=True).start()
        
    def _worker(self):
        """Worker thread to process faces"""
        try:
            # Only import DeepFace when needed to save startup time
            from deepface import DeepFace
            
            self.is_analyzing = True
            while True:
                if self.queue:
                    face_img, user_id = self.queue.pop(0)
                    try:
                        result = DeepFace.analyze(face_img, actions=['emotion'], enforce_detection=False)
                        if result:
                            emotion = result[0]['dominant_emotion']
                            self.results[user_id] = emotion
                    except Exception as e:
                        print(f"Error analyzing face: {str(e)}")
                else:
                    time.sleep(0.1)
        except Exception as e:
            print(f"Error in emotion analysis worker: {str(e)}")
            self.is_analyzing = False

# Initialize the async analyzer
face_analyzer = AsyncFaceAnalyzer()

def get_simple_color(image, face_location):
    """Simplified method to detect dominant clothing color"""
    top, right, bottom, left = face_location
    face_height = bottom - top
    
    # Define clothing area below face
    clothes_top = min(bottom + int(face_height * 0.2), image.shape[0] - 10)
    clothes_bottom = min(clothes_top + int(face_height), image.shape[0])
    clothes_left = max(0, left - int((right - left) * 0.2))
    clothes_right = min(right + int((right - left) * 0.2), image.shape[1])
    
    # Avoid invalid regions
    if clothes_bottom <= clothes_top or clothes_right <= clothes_left:
        return "unknown"
    
    # Extract region and calculate average color
    clothes_region = image[clothes_top:clothes_bottom, clothes_left:clothes_right]
    average_color = np.mean(clothes_region, axis=(0, 1))
    
    # Simple color classification based on BGR values
    b, g, r = average_color
    
    # Determine if color is dark or light
    brightness = (r + g + b) / 3
    brightness_label = "dark" if brightness < 85 else "light" if brightness > 170 else ""
    
    # Simplified color detection using relative channel values
    if b > r and b > g:
        color = "blue"
    elif g > r and g > b:
        color = "green"
    elif r > g and r > b:
        if g > b + 50:
            color = "orange" if g > 100 else "brown"
        else:
            color = "red"
    elif abs(r - g) < 20 and abs(r - b) < 20 and abs(g - b) < 20:
        if brightness < 60:
            color = "black"
        elif brightness > 200:
            color = "white"
        else:
            color = "gray"
    else:
        color = "unknown"
    
    return f"{brightness_label} {color}".strip()

def get_time_based_greeting():
    """Return a greeting based on the current time of day."""
    current_hour = datetime.datetime.now().hour
    
    if 5 <= current_hour < 12:
        return "Good morning"
    elif 12 <= current_hour < 17:
        return "Good afternoon"
    elif 17 <= current_hour < 22:
        return "Good evening"
    else:
        return "Hello"

def load_known_faces():
    """Load all known face encodings from the dataset."""
    global known_face_encodings, known_face_names
    known_face_encodings = []
    known_face_names = []
    
    print("Loading face data...")
    for user in os.listdir(dataset_path):
        user_folder = os.path.join(dataset_path, user)
        if not os.path.isdir(user_folder):
            continue
        
        # Use only a few images per person for faster loading
        image_files = [f for f in os.listdir(user_folder) if f.endswith(('.jpg', '.png'))]
        # Take only first 5 images for faster processing
        for image_name in image_files[:5]:
            image_path = os.path.join(user_folder, image_name)
            try:
                image = face_recognition.load_image_file(image_path)
                # Use faster detection model (HOG instead of CNN)
                face_locations = face_recognition.face_locations(image, model="hog")
                
                if face_locations:
                    encoding = face_recognition.face_encodings(image, [face_locations[0]])[0]
                    known_face_encodings.append(encoding)
                    known_face_names.append(user)
                    print(f"Loaded face data: {user} - {image_name}")
            except Exception as e:
                print(f"Error loading {image_path}: {str(e)}")
    
    print(f"Loaded {len(known_face_encodings)} face encodings for {len(set(known_face_names))} users")

def load_user_visits():
    """Load the user visits data."""
    visits_file = os.path.join(visits_data_path, "user_visits.json")
    if os.path.exists(visits_file):
        try:
            with open(visits_file, 'r') as f:
                return json.load(f)
        except:
            return {}
    return {}

def save_user_visit(user_name):
    """Save the user visit to the visits data."""
    visits_data = load_user_visits()
    current_time = datetime.datetime.now().isoformat()
    
    if user_name not in visits_data:
        visits_data[user_name] = {
            "first_seen": current_time,
            "visits": []
        }
    
    # Add this visit
    visits_data[user_name]["visits"].append(current_time)
    visits_data[user_name]["last_seen"] = current_time
    
    # Keep only last 50 visits to save space
    if len(visits_data[user_name]["visits"]) > 50:
        visits_data[user_name]["visits"] = visits_data[user_name]["visits"][-50:]
    
    # Save to file
    visits_file = os.path.join(visits_data_path, "user_visits.json")
    with open(visits_file, 'w') as f:
        json.dump(visits_data, f, indent=4)

def get_user_visit_frequency(user_name):
    """Get the visit frequency status for a user."""
    visits_data = load_user_visits()
    
    if user_name not in visits_data:
        return "New visitor"
    
    visits = visits_data[user_name]["visits"]
    
    # First time seeing this user
    if len(visits) == 1:
        return "First visit"
    
    # Convert visit timestamps to datetime objects
    visit_dates = [datetime.datetime.fromisoformat(v) for v in visits]
    
    # Calculate how many visits in the last 7 days
    now = datetime.datetime.now()
    week_ago = now - datetime.timedelta(days=7)
    visits_this_week = sum(1 for date in visit_dates if date >= week_ago)
    
    # Calculate time since last visit
    last_visit = datetime.datetime.fromisoformat(visits_data[user_name]["last_seen"])
    days_since_last = (now - last_visit).days
    
    if visits_this_week >= 5:
        return "Frequent visitor"
    elif 2 <= visits_this_week < 5:
        return "Regular visitor"
    elif days_since_last > 14:
        return "Returning after a while"
    else:
        return "Occasional visitor"

def home_page():
    """Create the home page."""
    from tkinter import Tk, Label, Button, Entry, StringVar, Frame, Listbox, Scrollbar, VERTICAL, RIGHT, Y, END
    
    home = Tk()
    home.title("Face Recognizer")
    home.geometry("400x400")
    Label(home, text="Face Recognizer", font=("Arial", 20)).pack(pady=20)
    Button(home, text="Add a User", width=20, command=lambda: add_user_page(home)).pack(pady=10)
    Button(home, text="Check a User", width=20, command=lambda: check_user_page(home)).pack(pady=10)
    Button(home, text="View Visit History", width=20, command=lambda: view_visits_page(home)).pack(pady=10)
    Button(home, text="Exit", width=20, command=home.destroy).pack(pady=10)
    home.mainloop()

def add_user_page(home):
    """Create the add user page."""
    from tkinter import Tk, Label, Button, Entry, StringVar
    
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
    from tkinter import Tk, Label, Button
    global user_name, images_captured
    
    if not name.strip():
        from tkinter import messagebox
        messagebox.showerror("Error", "Please enter a valid name")
        return
    
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

    # Use HOG-based face detector for better performance
    frame_count = 0
    
    while images_captured < 15:  # Reduced number of images
        ret, frame = cap.read()
        if not ret:
            break
            
        frame_count += 1
        if frame_count % 2 != 0:  # Process every other frame
            continue
            
        # Use face_recognition's HOG detector which is faster
        face_locations = face_recognition.face_locations(frame, model="hog")
        
        for face_location in face_locations:
            top, right, bottom, left = face_location
            face_width = right - left
            face_height = bottom - top
            
            # Only process faces of reasonable size
            if face_width < 100 or face_height < 100:
                continue
                
            # Create a zoomed face region with some margin
            margin = 20
            face_img = frame[
                max(0, top-margin):min(frame.shape[0], bottom+margin),
                max(0, left-margin):min(frame.shape[1], right+margin)
            ]
            
            # Save image
            image_path = os.path.join(user_folder, f"{images_captured}.jpg")
            cv2.imwrite(image_path, face_img)
            
            # Draw rectangle
            cv2.rectangle(frame, (left, top), (right, bottom), (0, 255, 0), 2)
            
            images_captured += 1
            count_label.config(text=f"Images captured: {images_captured}")
            
            # Show a message on screen
            cv2.putText(frame, f"Captured {images_captured}/15", 
                       (left, top-10), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
            
            # Break after capturing one face per frame
            break
            
        cv2.imshow("Capture Images", frame)
        if cv2.waitKey(100) & 0xFF == ord('q') or images_captured >= 15:
            break

    cap.release()
    cv2.destroyAllWindows()
    
    engine.say(f"Dataset captured for {user_name}. Please click Train Model to continue.")
    engine.runAndWait()

def train_model():
    """Train the face recognition model."""
    # Load the known faces
    load_known_faces()
    
    engine.say("Training completed successfully")
    engine.runAndWait()

def check_user_page(home):
    """Create the check user page."""
    from tkinter import Tk, Label, Button
    
    home.destroy()
    check_user = Tk()
    check_user.title("Check a User")
    check_user.geometry("400x300")
    Label(check_user, text="Check User", font=("Arial", 14)).pack(pady=10)
    Button(check_user, text="Recognize User", width=20, command=recognize_user).pack(pady=10)
    Button(check_user, text="Home", width=20, command=lambda: (check_user.destroy(), home_page())).pack(pady=10)
    check_user.mainloop()

def view_visits_page(home):
    """Create the view visits page."""
    from tkinter import Tk, Label, Button, Frame, Listbox, Scrollbar, VERTICAL, RIGHT, Y, END
    
    home.destroy()
    visits_page = Tk()
    visits_page.title("View Visit History")
    visits_page.geometry("600x500")
    
    Label(visits_page, text="User Visit History", font=("Arial", 18)).pack(pady=10)
    
    # Create a frame for the listbox and scrollbar
    frame = Frame(visits_page)
    frame.pack(pady=10, fill="both", expand=True)
    
    # Create scrollbar
    scrollbar = Scrollbar(frame, orient=VERTICAL)
    scrollbar.pack(side=RIGHT, fill=Y)
    
    # Create listbox
    visit_listbox = Listbox(frame, width=70, height=20, font=("Arial", 12))
    visit_listbox.pack(pady=10, fill="both", expand=True)
    
    # Configure scrollbar
    scrollbar.config(command=visit_listbox.yview)
    visit_listbox.config(yscrollcommand=scrollbar.set)
    
    # Load visits data
    visits_data = load_user_visits()
    
    if not visits_data:
        visit_listbox.insert(END, "No visit history found")
    else:
        visit_listbox.insert(END, f"{'User':<20} {'Last Visit':<30} {'Visit Count':<10} {'Status'}")
        visit_listbox.insert(END, "-" * 70)
        
        for user, data in visits_data.items():
            last_visit = datetime.datetime.fromisoformat(data["last_seen"]).strftime("%Y-%m-%d %H:%M")
            visit_count = len(data["visits"])
            status = get_user_visit_frequency(user)
            
            visit_listbox.insert(END, f"{user:<20} {last_visit:<30} {visit_count:<10} {status}")
    
    Button(visits_page, text="Home", width=20, command=lambda: (visits_page.destroy(), home_page())).pack(pady=10)
    visits_page.mainloop()

# Function to create an announcement card with visitor info
def create_announcement_card(base_frame, person_name, visit_status, emotion, clothing_color):
    """Create a visually appealing announcement card with person info"""
    h, w = base_frame.shape[:2]
    
    # Create a semi-transparent card overlay
    card = base_frame.copy()
    overlay = np.zeros_like(card)
    
    # Card dimensions and position
    card_width = int(w * 0.8)
    card_height = 150
    start_x = (w - card_width) // 2
    start_y = h - card_height - 50  # Position from bottom with some margin
    
    # Create card background (semi-transparent blue)
    cv2.rectangle(overlay, (start_x, start_y), (start_x + card_width, start_y + card_height), 
                 (200, 100, 40), -1)
    
    # Create header bar
    header_height = 40
    cv2.rectangle(overlay, (start_x, start_y), (start_x + card_width, start_y + header_height), 
                 (40, 40, 200), -1)
    
    # Add borders
    cv2.rectangle(overlay, (start_x, start_y), (start_x + card_width, start_y + card_height), 
                 (255, 255, 255), 2)
    
    # Blend overlay with original frame
    cv2.addWeighted(overlay, 0.7, card, 1.0, 0, card)
    
    # Add header text
    text = "VISITOR DETECTED"
    text_size = cv2.getTextSize(text, cv2.FONT_HERSHEY_DUPLEX, 0.8, 2)[0]
    text_x = start_x + (card_width - text_size[0]) // 2
    cv2.putText(card, text, (text_x, start_y + 30), 
               cv2.FONT_HERSHEY_DUPLEX, 0.8, (255, 255, 255), 2)
    
    # Add visitor information with bold, easy-to-read formatting
    y_pos = start_y + header_height + 30
    
    # Name and status
    name_text = f"Name: {person_name}"
    cv2.putText(card, name_text, (start_x + 20, y_pos), 
               cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
    
    status_text = f"Status: {visit_status}"
    cv2.putText(card, status_text, (start_x + card_width - 250, y_pos), 
               cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
    
    # Emotion and clothing
    y_pos += 40
    emotion_text = f"Mood: {emotion}"
    cv2.putText(card, emotion_text, (start_x + 20, y_pos), 
               cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
    
    clothing_text = f"Clothing: {clothing_color}"
    cv2.putText(card, clothing_text, (start_x + card_width - 250, y_pos), 
               cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
    
    return card

# Alias function for backward compatibility
def create_info_panel(frame, active_announcements):
    """Create info panel with active announcements (alias for create_announcement_card)"""
    if active_announcements:
        latest_name = list(active_announcements.keys())[-1]
        info = active_announcements[latest_name]
        return create_announcement_card(
            frame, 
            latest_name,
            info['visit_status'],
            info['emotion'],
            info['clothing']
        )
    return frame

def recognize_user():
    """Recognize the user from the webcam."""
    if not known_face_encodings:
        load_known_faces()
        if not known_face_encodings:
            print("No face data available")
            engine.say("No users found in database")
            engine.runAndWait()
            return

    # Start emotion analyzer thread
    if not face_analyzer.is_analyzing:
        face_analyzer.start_worker()

    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("Error: Cannot open camera")
        return

    # Use deque to manage stable detection
    face_memory = {}  # Track detections for each user
    greeted_users = set()  # Track who has been greeted
    speech_cooldown = 0
    frame_count = 0
    
    # Dictionary to store active announcements
    active_announcements = {}
    announcement_duration = 150  # frames (about 5 seconds at 30fps)
    
    while True:
        ret, frame = cap.read()  # FIXED: Changed 'map' to 'cap'
        if not ret:
            break

        frame_count += 1
        
        # Display current time and date at the top
        current_datetime = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        cv2.putText(frame, current_datetime, (10, 30), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
                   
        # Process every 3rd frame for better performance
        should_process = (frame_count % 3 == 0)
        
        # Always display latest visitor cards even on non-processed frames
        display_frame = frame.copy()
        
        # Display active announcement (only shows the most recent one)
        if active_announcements:
            latest_name = list(active_announcements.keys())[-1]
            info = active_announcements[latest_name]
            display_frame = create_announcement_card(
                display_frame, 
                latest_name,
                info['visit_status'],
                info['emotion'],
                info['clothing']
            )
            
        if not should_process:
            # Display but don't process further
            cv2.imshow("Face Recognition", display_frame)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
            continue

        # Detect faces using HOG (faster than CNN)
        face_locations = face_recognition.face_locations(frame, model="hog")
        
        current_faces = set()  # Track who's in current frame
        
        if face_locations:
            # Only compute encodings if faces are detected (saves CPU)
            face_encodings = face_recognition.face_encodings(frame, face_locations)

            for face_location, face_encoding in zip(face_locations, face_encodings):
                # Use a higher tolerance for better matching (default is 0.6)
                matches = face_recognition.compare_faces(known_face_encodings, face_encoding, tolerance=0.55)
                
                # Only calculate distances if we have matches (saves CPU)
                if any(matches):
                    face_distances = face_recognition.face_distance(known_face_encodings, face_encoding)
                    best_match_index = np.argmin(face_distances)
                    
                    # Get the name if it's a good match
                    name = "Unknown"
                    if matches[best_match_index]:
                        name = known_face_names[best_match_index]
                        current_faces.add(name)
                        
                        # Initialize tracking data for this person
                        if name not in face_memory:
                            face_memory[name] = {
                                "count": 0,
                                "clothing_color": None,
                                "last_seen": time.time(),
                                "visit_status": get_user_visit_frequency(name)
                            }
                        
                        # Update detection count
                        face_memory[name]["count"] += 1
                        face_memory[name]["last_seen"] = time.time()
                        
                        # Get clothing color if we don't have it yet
                        if face_memory[name]["clothing_color"] is None and face_memory[name]["count"] > 5:
                            color = get_simple_color(frame, face_location)
                            face_memory[name]["clothing_color"] = color
                        
                        # Queue for emotion analysis if we have enough stable detections
                        if face_memory[name]["count"] > 10:
                            top, right, bottom, left = face_location
                            face_img = frame[top:bottom, left:right]
                            face_analyzer.analyze_emotion(face_img, name)
                        
                        # Process a stable detection
                        if (face_memory[name]["count"] > 15 and 
                            speech_cooldown <= 0 and
                            name not in greeted_users):
                            
                            # Record this visit
                            save_user_visit(name)
                            
                            # Get information for greeting
                            time_greeting = get_time_based_greeting()
                            clothing_color = face_memory[name]["clothing_color"] or "unknown"
                            visit_status = face_memory[name]["visit_status"]
                            emotion = face_analyzer.results.get(name, "neutral")
                            
                            # Prepare greeting
                            gender = "his" if name.lower().endswith(('o', 'n', 'm', 'l', 'k', 'd', 'b')) else "her"
                            greeting = f"{time_greeting}, {name} is in front of you. {gender} mood is {emotion}. {gender.capitalize()} is wearing {clothing_color} clothes. {gender.capitalize()} is a {visit_status}."
                            
                            # Speak the greeting
                            Thread(target=lambda: engine.say(greeting) or engine.runAndWait()).start()
                            
                            # Store greeting info for visual display
                            active_announcements[name] = {
                                "visit_status": visit_status,
                                "emotion": emotion,
                                "clothing": clothing_color,
                                "frames_left": announcement_duration
                            }
                            
                            # Mark as greeted and set cooldown
                            greeted_users.add(name)
                            speech_cooldown = 50  # frames
                
                # Draw rectangles for face
                top, right, bottom, left = face_location
                cv2.rectangle(display_frame, (left, top), (right, bottom), (0, 255, 0), 2)
                
                # Display info
                if name != "Unknown":
                    visit_status = face_memory[name]["visit_status"] if name in face_memory else ""
                    emotion = face_analyzer.results.get(name, "analyzing...")
                    cv2.putText(display_frame, f"{name} - {visit_status}", (left, top-10), 
                              cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
                    cv2.putText(display_frame, f"Emotion: {emotion}", (left, top-35), 
                              cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)
                    
                    # Show clothing info if available
                    if name in face_memory and face_memory[name]["clothing_color"]:
                        cv2.putText(display_frame, f"Clothes: {face_memory[name]['clothing_color']}", 
                                  (left, bottom+25), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 0, 0), 2)
                else:
                    cv2.putText(display_frame, "Unknown", (left, top-10), 
                              cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)
        
        # Clean up old entries (no longer detected)
        current_time = time.time()
        remove_keys = []
        for name in face_memory:
            if name not in current_faces and (current_time - face_memory[name]["last_seen"]) > 5:
                remove_keys.append(name)
        
        for name in remove_keys:
            del face_memory[name]
        
        # Decrement speech cooldown
        if speech_cooldown > 0:
            speech_cooldown -= 1
            
        # Update announcement durations and remove expired ones
        # Decrement speech cooldown
       
        # Update announcement durations and remove expired ones
        expired_announcements = []
        for name in active_announcements:
            active_announcements[name]["frames_left"] -= 1
            if active_announcements[name]["frames_left"] <= 0:
                expired_announcements.append(name)
                
        for name in expired_announcements:
            del active_announcements[name]
        
        # Add the info panel if we have active announcements
        if active_announcements:
            frame = create_info_panel(frame, active_announcements)
        
        # Display frame
        cv2.imshow("Face Recognition", frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    home_page()