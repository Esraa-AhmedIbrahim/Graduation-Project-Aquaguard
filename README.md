🎓 AquaGuard – AI-Powered Beach Safety System
Graduation project developed at Ain Shams University (2025), awarded an A+ grade
AquaGuard is an intelligent beach safety system that leverages drone surveillance, deep learning, and computer vision to detect and alert for three major coastal hazards:
🏊‍♂ Drowning incidents
🦈 Shark presence
🌊 Rip currents

#Key Modules & Technologies
1. Drowning Detection
Uses transformer-based video classifiers to recognize potential drowning behavior
Employs DeepSORT and ConvNeXt-Tiny for real-time person tracking
2. Shark & Human Detection
Utilizes YOLOv11 for object detection in aquatic environments
3. Rip Current Classification
Fine-tuned ResNet-50 model to classify rip currents from aerial footage
4. Integrated Alert System
Backend built with FastAPI, optimized for GPU acceleration
Mobile alerts delivered via a Flutter-based app to notify lifeguards instantly

#How AquaGuard Works
1) Data Source – Drone Footage
A drone captures aerial footage of the beach, which is used as input.
2) Preprocessing
The video is split into frames and prepared for analysis by AI models.
AI Models – Logic Layer
3) Frame-Based Detection:
Rip Current Detection: ResNet processes frames to detect dangerous currents.
Shark & Human Detection: YOLO detects humans and sharks in the same frame.
4) Sequence-Based Detection:
If a human is detected, their movement is tracked using DeepSORT.
A Transformer model analyzes the sequence of frames to detect drowning behavior.
5) Alert Generation & Severity Handling
If a threat is detected, the system generates an alert.
6) Severity Level:
High severity: When a human is detected near a shark or inside a rip current.
Low severity: When the threat is present without a human nearby .
7) User Notification & Data Storage
Alerts with severity levels are sent to registered users via the Flutter mobile app,
while user and alert data are stored and managed using Firebase services

Link for Demo : https://drive.google.com/file/d/1DVGoBSFlTZHCpIUwSdlCTr68lTSnQtWq/view?usp=sharing

#Development and Design by 
1) Hanin Ayman El Sayed
2) Esraa Ahmed Ibrahim
3) Rahma Sha3ban Esmail
4) Mohamed Emad Mohamed
5) Ali Khalid Mahmoud
5) Mostafa Saad

#Academic Information
Institution: Ain Shams University – Faculty of Computer and Information Science.
Project Type: Graduation Project (2025)
Grade: A+
Supervisor: Dr. Ahmed Salah, Department of Computer Science.
