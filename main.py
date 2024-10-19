import cv2
import numpy as np
import pandas as pd
import datetime
import pytz
import time
import os
from twilio.rest import Client

# Global variables
last_sos_time = 0
sos_cooldown = 30  # Cooldown period in seconds
hotspot_threshold = 5  # Minimum number of SOS alerts for hotspot detection
hotspot_duration_days = 10  # Time window to track SOS alerts (days)

# Initialize models
print("Initializing models...")
try:
    face_net = cv2.dnn.readNetFromCaffe('C:/SIH/TheReal/deploy.prototxt', 'C:/SIH/TheReal/res10_300x300_ssd_iter_140000_fp16.caffemodel')
    gender_net = cv2.dnn.readNetFromCaffe('C:/SIH/TheReal/gender_deploy.prototxt', 'C:/SIH/TheReal/gender_net.caffemodel')
except Exception as e:
    print(f"Error initializing models: {e}")
    exit()

# Twilio configuration
print("Configuring Twilio...")
account_sid = 'AC913382270036261b1fa27480a6b942f4'
auth_token = '3399a9b59d40498578ba1b5a76c7f7d1'
twilio_phone_number = '+16318503785'
your_phone_number = '+918421912658'
client = Client(account_sid, auth_token)

# Incident DataFrame
incident_df = pd.DataFrame(columns=['datetime', 'location'])

def detect_person_gender(frame):
    height, width = frame.shape[:2]
    people_detected = {'men': 0, 'women': 0}

    # Detect faces
    blob = cv2.dnn.blobFromImage(cv2.resize(frame, (300, 300)), 1.0, (300, 300), (104.0, 177.0, 123.0))
    face_net.setInput(blob)
    detections = face_net.forward()

    for i in range(detections.shape[2]):
        confidence = detections[0, 0, i, 2]
        if confidence > 0.5:
            box = detections[0, 0, i, 3:7] * np.array([width, height, width, height])
            (startX, startY, endX, endY) = box.astype("int")

            face = frame[max(0, startY):min(height, endY), max(0, startX):min(width, endX)]
            if face.size == 0:
                continue

            face_blob = cv2.dnn.blobFromImage(cv2.resize(face, (227, 227)), 1.0, (227, 227), (104.0, 177.0, 123.0))
            gender_net.setInput(face_blob)
            gender_detections = gender_net.forward()

            gender = "men" if gender_detections[0][0] > gender_detections[0][1] else "women"
            people_detected[gender] += 1

    return people_detected

def is_night():
    tz = pytz.timezone('Asia/Kolkata')
    current_time = datetime.datetime.now(tz).time()
    return current_time.hour < 6 or current_time.hour > 18

def detect_sos_gesture(frame):
    # Placeholder for SOS gesture detection
    return False

def log_sos_event(location):
    """Logs the SOS alert in a CSV file with date, time, and location."""
    current_time = datetime.datetime.now()
    incident_df.loc[len(incident_df)] = [current_time, location]
    incident_df.to_csv('C:/SIH/TheReal/incidents.csv', index=False)
    print(f"SOS event logged for location: {location}")

def send_sos_alert(location):
    global last_sos_time
    current_time = time.time()
    if current_time - last_sos_time >= sos_cooldown:
        print("Sending SOS alert...")
        client.messages.create(
            body=f"SOS Alert! Immediate attention required at {location}.",
            from_=twilio_phone_number,
            to=your_phone_number
        )
        print("SOS alert sent!")
        last_sos_time = current_time
        log_sos_event(location)
    else:
        print("SOS alert cooldown period active. Skipping alert.")

def process_frame(frame, location="Current Location"):
    print("Processing frame...")
    people_detected = detect_person_gender(frame)
    
    men_count = people_detected['men']
    women_count = people_detected['women']
    
    print(f"Men detected: {men_count}, Women detected: {women_count}")

    # Check if SOS condition based on gender counts
    if women_count < men_count:
        print("Fewer women than men detected.")
        send_sos_alert(location)

    # Nighttime detection
    if is_night():
        if women_count > 0 and men_count == 0:
            print("Lone woman detected at night.")
            send_sos_alert(location)

    # SOS Gesture Detection
    if detect_sos_gesture(frame):
        print("SOS gesture detected")
        send_sos_alert(location)

def identify_hotspots():
    """Analyzes SOS data and identifies hotspot areas based on frequent alerts."""
    current_time = datetime.datetime.now()
    hotspot_areas = {}

    # Filter incidents within the last `hotspot_duration_days`
    incident_df['datetime'] = pd.to_datetime(incident_df['datetime'])
    recent_incidents = incident_df[incident_df['datetime'] > current_time - pd.to_timedelta(hotspot_duration_days, unit='d')]

    # Count incidents by location
    location_counts = recent_incidents['location'].value_counts()

    # Identify hotspots based on the threshold
    for location, count in location_counts.items():
        if count >= hotspot_threshold:
            hotspot_areas[location] = count

    print("Hotspot areas identified:", hotspot_areas)
    return hotspot_areas

def process_video(video_path, location="Current Location"):
    print(f"Processing video: {video_path}")
    cap = cv2.VideoCapture(video_path)
    
    if not cap.isOpened():
        print(f"Error: Unable to open video file {video_path}.")
        return

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            print(f"End of video or error reading frame in {video_path}.")
            break

        process_frame(frame, location)

        # Optional: Display the frame
        cv2.imshow('Frame', frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

def track_incident(location):
    """Tracks incidents and stores them in a CSV file."""
    try:
        incidents_df = pd.read_csv('C:/SIH/TheReal/incidents.csv')
    except FileNotFoundError:
        incidents_df = pd.DataFrame(columns=['datetime', 'location'])

    current_time = datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')
    new_incident = pd.DataFrame({'datetime': [current_time], 'location': [location]})

    incidents_df = pd.concat([incidents_df, new_incident], ignore_index=True)
    incidents_df.to_csv('C:/SIH/TheReal/incidents.csv', index=False)

    return incidents_df

def check_hotspot_area(incidents_df):
    incidents_df['datetime'] = pd.to_datetime(incidents_df['datetime'])
    current_time = datetime.datetime.now()
    past_incidents = incidents_df[incidents_df['datetime'] >= (current_time - datetime.timedelta(days=hotspot_duration_days))]
    location_counts = past_incidents['location'].value_counts()

    hotspots = location_counts[location_counts >= hotspot_threshold]
    if not hotspots.empty:
        print(f"Hotspot areas identified: {hotspots.to_dict()}")
    else:
        print("No hotspots identified at this time.")
    
    return hotspots

def main():
    video_folder = 'C:/SIH/TheReal/videos'
    video_files = [f for f in os.listdir(video_folder) if f.endswith('.mp4')]

    if not video_files:
        print("No video files found in the specified directory.")
        return

    for video_file in video_files:
        video_path = os.path.join(video_folder, video_file)
        process_video(video_path)

    # Identify hotspots after processing
    identify_hotspots()
    
    location = "Location A"
    incidents_df = track_incident(location)
    check_hotspot_area(incidents_df)

if __name__ == "__main__":
    main()
