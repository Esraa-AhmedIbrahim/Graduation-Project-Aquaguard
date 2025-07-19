from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field
from datetime import datetime
from typing import List
from ultralytics import YOLO
import cv2
import logging
import os
from enum import Enum
from uuid import UUID, uuid4
import torch
from fastapi import BackgroundTasks
from starlette.concurrency import run_in_threadpool
import asyncio
import time
from datetime import datetime
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
import matplotlib.animation as animation
from deep_sort_realtime.deepsort_tracker import DeepSort
import torchvision.transforms as T
from collections import defaultdict, deque
import glob
from PIL import Image
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import time
import shutil
import threading
import asyncio
from uuid import uuid4
from datetime import datetime
from fastapi import HTTPException
from collections import deque
import uuid


logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")


Rip_model = YOLO('Rip-best.pt')
Shark_model = YOLO('Shark-best.pt')
Human_model = YOLO('Human-best.pt')


reid_model_path = "reid_model.pth"
yolo_model_path = "best (1).pt"
transformer_model_path = "drowning-best.pth"
output_dir = "working/tracking_results/video_1"
output_video = "working/drowning_detection_video.mp4"
    



async def check_human(image_path: str, confidence_threshold: float = 0.5):
    try:
        if not os.path.exists(image_path):
            raise HTTPException(status_code=400, detail="Local file not found")

        results = Human_model(image_path)

        Human_detected = False
        confidences = []

        for result in results:
            boxes = result.boxes
            for box in boxes:
                confidence = box.conf.item()  
                if confidence >= confidence_threshold:  
                    Human_detected = True
                    confidences.append(confidence)  

        result_image = results[0].plot()  
        result_image = results[0].plot()  
        result_image = results[0].plot()  
        output_dir = "Humans"
        os.makedirs(output_dir, exist_ok=True)  

        output_image_path = os.path.join(output_dir, f"output_image_{uuid4().hex}.jpg")


        return {
            "human_detected": Human_detected,
            "confidences": confidences,
            "annotated_image_path": output_image_path
        }

    except Exception as e:
        logger.error(f"An error occurred: {e}")
        raise HTTPException(status_code=500, detail=str(e))


async def check_ripcurrent(image_path: str, confidence_threshold: float = 0.5):
    try:
        if not os.path.exists(image_path):
            raise HTTPException(status_code=400, detail="Local file not found")

        results = Rip_model(image_path)

        ripcurrent_detected = False
        confidences = []

        for result in results:
            boxes = result.boxes
            for box in boxes:
                confidence = box.conf.item()  
                if confidence >= confidence_threshold:  
                    ripcurrent_detected = True
                    confidences.append(confidence)  

        result_image = results[0].plot()  
        output_dir = "rip_Current"
        os.makedirs(output_dir, exist_ok=True)  

        output_image_path = os.path.join(output_dir, f"output_image_{uuid4().hex}.jpg")


        if ripcurrent_detected:
            logger.info("Rip current detected! Confidence scores: %s", confidences)
            cv2.imwrite(output_image_path, result_image)
        else:
            logger.info("No rip current detected.")

        return {
            "ripcurrent_detected": ripcurrent_detected,
            "confidences": confidences,
            "annotated_image_path": output_image_path
        }

    except Exception as e:
        logger.error(f"An error occurred: {e}")
        raise HTTPException(status_code=500, detail=str(e))


async def check_shark(image_path: str, confidence_threshold: float = 0.5):
    try:
        if not os.path.exists(image_path):
            raise HTTPException(status_code=400, detail="Local file not found")

        results = Shark_model(image_path)

        shark_detected = False
        confidences = []

        for result in results:
            boxes = result.boxes
            for box in boxes:
                confidence = box.conf.item() 
                if confidence >= confidence_threshold:  
                    shark_detected = True
                    confidences.append(confidence)  

        result_image = results[0].plot()  
        result_image = results[0].plot()  
        output_dir = "Sharks"
        os.makedirs(output_dir, exist_ok=True)  

        output_image_path = os.path.join(output_dir, f"output_image_{uuid4().hex}.jpg")

 

        if shark_detected:
            logger.info("Shark detected! Confidence scores: %s", confidences)
            cv2.imwrite(output_image_path, result_image)
        else:
            logger.info("No Shark detected.")

        return {
            "shark_detected": shark_detected,
            "confidences": confidences,
            "annotated_image_path": output_image_path
        }

    except Exception as e:
        logger.error(f"An error occurred: {e}")
        raise HTTPException(status_code=500, detail=str(e))

app = FastAPI()
All_Detections = []
All_Alerts = []
Drowning_Alerts = []
Shark_Alerts = []
RipCurrent_Alerts = []


class Coordinates(BaseModel):
    lat: float
    lng: float


class DetectionType(Enum):
    Drowning = "Drowning"
    Sharks = "Shark"
    Rip_Current = "Rip Current"


class Location(BaseModel):
    beach: str
    coordinates: Coordinates


class Metadata(BaseModel):
    cameraId: str
    weatherConditions: str
    responseTeamDispatched: bool = False  


class DetectionRequest(BaseModel):
    location: Location
    videoPath: str
    metadata: Metadata
    timestamp: datetime = Field(default_factory=datetime.now)

class Shark(BaseModel):
    id: UUID = Field(default_factory=uuid4)  
    type: DetectionType
    confidence: float
    location: Location
    severity : str
    timestamp: datetime = datetime.now()
    imageUrl: str
    metadata: Metadata


class RipCurrent(BaseModel):
    id: UUID = Field(default_factory=uuid4)  
    type: DetectionType
    confidence: float
    location: Location
    severity : str
    timestamp: datetime = datetime.now()
    imageUrl: str
    metadata: Metadata

    
class Drowning(BaseModel):
    id: UUID = Field(default_factory=uuid4) 
    type: DetectionType
    confidence: float
    location: Location
    severity : str
    timestamp: datetime = datetime.now()
    imageUrl: str
    metadata: Metadata


FRAME_RATE = 5               
SEQUENCE_SECONDS = 5         
SEQUENCE_LENGTH = FRAME_RATE * SEQUENCE_SECONDS  

logging.basicConfig(level=logging.INFO)



FRAMES_PER_SEQUENCE = 30  

def split_video_into_sequences(video_path, base_frames_dir):
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise HTTPException(status_code=400, detail="Cannot open video.")

    os.makedirs(base_frames_dir, exist_ok=True)
    fps = cap.get(cv2.CAP_PROP_FPS)
    sample_interval = max(1, int(fps // FRAME_RATE))
    current_frame_idx = 0
    total_frame_count = 0
    sequence_index = 0
    sequence_frame_count = 0

    sequence_dir = os.path.join(base_frames_dir, f"sequence_{sequence_index:03d}")
    os.makedirs(sequence_dir, exist_ok=True)

    logging.info(f"Video FPS: {fps}, Sampling every {sample_interval} frames")

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        if current_frame_idx % sample_interval == 0:
            frame_path = os.path.join(sequence_dir, f"frame_{sequence_frame_count:05d}.jpg")
            cv2.imwrite(frame_path, frame)
            logging.info(f"Saved: {frame_path}")

            sequence_frame_count += 1
            total_frame_count += 1

            if sequence_frame_count >= FRAMES_PER_SEQUENCE:
                sequence_index += 1
                sequence_frame_count = 0
                sequence_dir = os.path.join(base_frames_dir, f"sequence_{sequence_index:03d}")
                os.makedirs(sequence_dir, exist_ok=True)

        current_frame_idx += 1

    cap.release()
    logging.info(f"Total frames saved: {total_frame_count}")


def drowning_detection_process(detection: DetectionRequest):
    All_Detections.append(detection)
    severity = "High"

    has_humans = False
    
    cap = cv2.VideoCapture(detection.videoPath)
    if not cap.isOpened():
        raise HTTPException(status_code=400, detail="Cannot open video.")
    
    fps = cap.get(cv2.CAP_PROP_FPS)
    frame_interval = int(fps / FRAME_RATE)
    
    while True:
        ret, frame = cap.read()
        if not ret:
            break
            
        if int(cap.get(cv2.CAP_PROP_POS_FRAMES)) % frame_interval == 0:
            temp_img_path = f"temp_frame_{uuid4().hex}.jpg"
            cv2.imwrite(temp_img_path, frame)
            human_result = asyncio.run(check_human(temp_img_path))
            os.remove(temp_img_path)
            
            if human_result["human_detected"]:
                has_humans = True
                break
                
    cap.release()
    
    if not has_humans:
        logging.info("No humans detected in video, skipping drowning detection")
        return False

    frames_base_dir = f"working/sequences/{uuid.uuid4()}"
    os.makedirs(frames_base_dir, exist_ok=True)

    split_video_into_sequences(detection.videoPath, frames_base_dir)

    reid_model_path = "reid_model.pth"
    yolo_model_path = "best (1).pt"
    transformer_model_path = "drowning-best.pth"
    output_base_dir = "working/tracking_results"
    output_video = "working/drowning_detection_video.mp4"

    results_all = []

    sequence_dirs = sorted(glob.glob(os.path.join(frames_base_dir, "sequence_*")))
    for i, sequence_dir in enumerate(sequence_dirs):
        logging.info(f"Processing sequence {i}: {sequence_dir}")

        output_dir = os.path.join(output_base_dir, f"video_seq_{i}")
    
        results = process_video_for_drowning_detection(
            frames_dir=sequence_dir,
            reid_model_path=reid_model_path,
            yolo_model_path=yolo_model_path,
            transformer_model_path=transformer_model_path,
            output_dir=output_dir,
            output_video=output_video,
            frame_limit=None
        )

        results_all.append(results)

    print("########DROWNING############")

    Drowning_Alerts = []
    index = 0

    for result in results_all:
        if 'drowning_predictions' not in result:
            continue

        for key, predictions in result['drowning_predictions'].items():
            for prediction in predictions:
                if index % 6 == 0:
                    Drowning_Alerts.append(
                        Drowning(
                            type=DetectionType.Drowning,
                            confidence=prediction['confidence'],
                            location=detection.location,
                            timestamp=datetime.now(),
                            imageUrl="",
                            metadata=detection.metadata,
                            severity=severity
                        )
                    )
                    All_Alerts.append(
                        Drowning(
                            type=DetectionType.Drowning,
                            confidence=prediction['confidence'],
                            location=detection.location,
                            timestamp=datetime.now(),
                            imageUrl="",
                            metadata=detection.metadata,
                            severity=severity
                        )
                    )
                index += 1

    return bool(Drowning_Alerts)



def process_detection(detection: DetectionRequest):
    All_Detections.append(detection)
    severity = "Low"

    cap = cv2.VideoCapture(detection.videoPath)
    if not cap.isOpened():
        raise HTTPException(status_code=400, detail="Cannot open video.")

    fps = cap.get(cv2.CAP_PROP_FPS)
    frame_interval = int(fps / FRAME_RATE)

    
    last_rip_alert_time = -float('inf')
    last_shark_alert_time = -float('inf')

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        
        if int(cap.get(cv2.CAP_PROP_POS_FRAMES)) % frame_interval == 0:
            
            current_video_time = cap.get(cv2.CAP_PROP_POS_MSEC) / 1000.0

            
            temp_img_path = f"temp_frame_{uuid4().hex}.jpg"
            cv2.imwrite(temp_img_path, frame)

            human_result     = asyncio.run(check_human(temp_img_path))
            ripcurrent_result = asyncio.run(check_ripcurrent(temp_img_path))
            shark_result     = asyncio.run(check_shark(temp_img_path))
            os.remove(temp_img_path)
           
            if human_result["human_detected"]:
                severity = "High"


            if ripcurrent_result["ripcurrent_detected"]:
                
                if current_video_time - last_rip_alert_time >= 8.0:
                    rip_alert = RipCurrent(
                        type=DetectionType.Rip_Current,
                        confidence=max(ripcurrent_result["confidences"]),
                        location=detection.location,
                        timestamp=datetime.now(),
                        imageUrl=ripcurrent_result["annotated_image_path"],
                        metadata=detection.metadata,
                        severity=severity
                    )
                    RipCurrent_Alerts.append(rip_alert)
                    All_Alerts.append(rip_alert)
                    last_rip_alert_time = current_video_time

           
            if shark_result["shark_detected"]:
                if current_video_time - last_shark_alert_time >= 8.0:
                    shark_alert = Shark(
                        type=DetectionType.Sharks,
                        confidence=max(shark_result["confidences"]),
                        location=detection.location,
                        timestamp=datetime.now(),
                        imageUrl=shark_result["annotated_image_path"],
                        metadata=detection.metadata,
                        severity=severity
                    )
                    Shark_Alerts.append(shark_alert)
                    All_Alerts.append(shark_alert)
                    last_shark_alert_time = current_video_time

    cap.release()
    return {True}


@app.post("/api/detections")
async def create_detection(detection: DetectionRequest):
    results  = await asyncio.gather(
        run_in_threadpool(process_detection, detection),
        run_in_threadpool(drowning_detection_process, detection)
    )

    if all(results):
        return {"message": "Detection processed successfully"}
    else:
        return {"message": "Detection failed in one or more processes", "details": results}


@app.get("/api/detections")
async def get_detections():
    return All_Detections


@app.get("/api/alerts")
async def get_alerts():
    return {
        "all_alerts": All_Alerts[::-1]
        
    }

@app.get("/api/rip-current")
async def get_ripCurrent():
    return {
        "Rip-Currents" : RipCurrent_Alerts
    }

@app.get("/api/sharks")
async def   get_sharks():
    return{
        "Sharks" : Shark_Alerts
    }
    
    



class LightweightDrowningDetectionTransformer(nn.Module):
    def __init__(self, feature_dim=256, nhead=4, num_layers=2, dropout=0.1, input_dim=128):
        super().__init__()
        
        self.feature_extractor = nn.Sequential(
            nn.Conv2d(3, 16, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),  
            nn.Conv2d(16, 32, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),  
            nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),  
            nn.Flatten(),  
            nn.Linear(64 * 8 * 8, feature_dim)
        )
        
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=feature_dim, 
            nhead=nhead, 
            dropout=dropout,
            batch_first=True
        )
        self.transformer_encoder = nn.TransformerEncoder(
            encoder_layer, 
            num_layers=num_layers
        )
        
        self.classification = nn.Sequential(
            nn.Linear(feature_dim, feature_dim//2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(feature_dim//2, 2)
        )
    
    def forward(self, x, attention_mask=None):
        batch_size, seq_length = x.shape[0], x.shape[1]
        
        sequence_features = x  
        
        if attention_mask is not None:
           
            mask = (1 - attention_mask).bool()
            transformed_sequence = self.transformer_encoder(
                sequence_features, 
                src_key_padding_mask=mask
            )
        else:
            transformed_sequence = self.transformer_encoder(sequence_features)
        
        pooled_features = transformed_sequence.mean(dim=1)
        return self.classification(pooled_features)

class ReIDModel(nn.Module):
    def __init__(self, num_classes):
        super().__init__()
        import torchvision.models as models
        
        resnet = models.resnet18(weights="IMAGENET1K_V1")
        self.backbone = nn.Sequential(*list(resnet.children())[:-2])  
        self.pool = nn.AdaptiveAvgPool2d((1, 1))
        self.embedding = nn.Linear(512, 128)
        self.classifier = nn.Linear(128, num_classes)

    def forward(self, x):
        x = self.backbone(x)
        x = self.pool(x).view(x.size(0), -1)
        feat = self.embedding(x)
        out = self.classifier(feat)
        return F.normalize(feat, dim=1), out  

def extract_features(reid_model, img_crop, transform, device):
    
    if img_crop is None or img_crop.size == 0:
        return None
    
    img_tensor = transform(img_crop).unsqueeze(0).to(device)
    
    with torch.no_grad():
        embedding, _ = reid_model(img_tensor)
        
    return embedding.cpu()

def load_and_prepare_models(reid_model_path, yolo_model_path, transformer_model_path=None, device=None):
    
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    print(f"Using device: {device}")
    
    yolo = YOLO(yolo_model_path)
    
    num_classes = 90  
    reid_model = ReIDModel(num_classes)
    reid_model.load_state_dict(torch.load(reid_model_path, map_location=device))
    reid_model.to(device)
    reid_model.eval()
    
    transform = T.Compose([
        T.ToPILImage(),
        T.Resize((128, 64)),
        T.ToTensor(),
        T.Normalize(mean=[0.485, 0.456, 0.406],
                    std=[0.229, 0.224, 0.225])
    ])
    
    tracker = DeepSort(
        max_age=30,
        n_init=3,
        nms_max_overlap=1.0,
        max_cosine_distance=0.2,
        nn_budget=100,
        override_track_class=None,
        embedder=None,  
        bgr=True,
    )
    
    transformer_model = None
    if transformer_model_path and os.path.exists(transformer_model_path):
        print(f"Loading transformer model from {transformer_model_path}")
        transformer_model = LightweightDrowningDetectionTransformer(
            feature_dim=128,  
            nhead=4,  
            num_layers=2,  
            dropout=0.3,
            input_dim=64  
        )
       
        try:
            transformer_model.load_state_dict(torch.load(transformer_model_path, map_location=device))
        except Exception as e:
            print(f"Warning: Could not load transformer model weights: {e}")
            print("Creating a new model instead.")
        
        transformer_model.to(device)
        transformer_model.eval()
    else:
        print("No transformer model provided or found.")
    
    return {
        'yolo': yolo,
        'reid_model': reid_model,
        'transform': transform,
        'tracker': tracker,
        'transformer_model': transformer_model,
        'device': device
    }


def process_video_for_drowning_detection(
    frames_dir, 
    reid_model_path, 
    yolo_model_path, 
    transformer_model_path=None,
    gt_file=None, 
    output_dir=None, 
    output_video=None,
    frame_limit=None,
    sequence_length=5,
    sequence_overlap=1
):
    
    models = load_and_prepare_models(reid_model_path, yolo_model_path, transformer_model_path)
    yolo = models['yolo']
    reid_model = models['reid_model']
    transform = models['transform']
    tracker = models['tracker']
    transformer_model = models['transformer_model']
    device = models['device']
    

    drowning_output_dir = os.path.join(output_dir, "Drowining") if output_dir else None
    if drowning_output_dir:
        os.makedirs(drowning_output_dir, exist_ok=True)


    frame_files = sorted(glob.glob(os.path.join(frames_dir, '*.jpg')))
    if not frame_files:
        print(f"Error: No frames found in {frames_dir}")
        return
    
    if frame_limit is not None:
        frame_files = frame_files[:frame_limit]
    
   
    if output_dir and not os.path.exists(output_dir):
        os.makedirs(output_dir)
    
    tracks_history = {}
    
    track_features = defaultdict(lambda: deque(maxlen=sequence_length))
    track_crops = defaultdict(lambda: deque(maxlen=sequence_length))
    
    drowning_predictions = defaultdict(list)
    
    print("Starting frame processing...")
    
    for frame_count, frame_file in enumerate(frame_files, 1):
        frame = cv2.imread(frame_file)
        if frame is None:
            print(f"Error: Could not read frame {frame_file}")
            continue
            
        height, width = frame.shape[:2]
        
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        
        results = yolo(rgb_frame)
        detections = []
        detection_crops = []
        
        for r in results:
            boxes = r.boxes
            for box in boxes:
                cls_id = int(box.cls.item())
                conf = box.conf.item()
                
                if cls_id == 0 and conf > 0.5:  
                    x1, y1, x2, y2 = map(int, box.xyxy[0].tolist())
                    
                    x1, y1 = max(0, x1), max(0, y1)
                    x2, y2 = min(width, x2), min(height, y2)
                    
                    w, h = x2 - x1, y2 - y1
                    
                    if w <= 0 or h <= 0:
                        continue
                    
                    crop = rgb_frame[y1:y2, x1:x2]
                    detection_crops.append(crop)
                    
                    detection_data = (x1, y1, w, h, conf)
                    detections.append(detection_data)
        
        valid_detections = []
        valid_embeddings = []
        valid_crops = []
        
        for i, (det, crop) in enumerate(zip(detections, detection_crops)):
            x1, y1, w, h, conf = det
            
            embedding = extract_features(reid_model, crop, transform, device)
            
            if embedding is not None:
                valid_detections.append(([x1, y1, w, h], conf, 'person'))
                
                flat_embedding = embedding.squeeze().numpy()
                valid_embeddings.append(flat_embedding)
                valid_crops.append(crop)
        
        tracks = tracker.update_tracks(valid_detections, embeds=valid_embeddings, frame=frame)
        
        tracks_history[frame_count] = tracks.copy()
        
        print(f"\nProcessing Frame {frame_count}:")
        if len(tracks) == 0:
            print("  No tracks detected in this frame")
        
        for i, track in enumerate(tracks):
            if track.is_confirmed():
                track_id = track.track_id
                
                det_idx = None
                for j, det in enumerate(valid_detections):
                    ltrb = track.to_ltrb()
                    x1, y1, x2, y2 = map(int, ltrb)
                    det_x1, det_y1, det_w, det_h = det[0]
                    det_x2, det_y2 = det_x1 + det_w, det_y1 + det_h
                    
                    iou = calculate_iou(
                        [x1, y1, x2, y2],
                        [det_x1, det_y1, det_x2, det_y2]
                    )
                    if iou > 0.5:
                        det_idx = j
                        break
                
                if det_idx is not None:
                    embedding = valid_embeddings[det_idx]
                    crop = valid_crops[det_idx]
                    
                    track_features[track_id].append(embedding)
                    track_crops[track_id].append(crop)
                    
                    if transformer_model is not None and len(track_features[track_id]) >= sequence_length:
                        features = np.array(list(track_features[track_id]))
                        features_tensor = torch.FloatTensor(features).unsqueeze(0).to(device)

                        
                        
                        with torch.no_grad():
                            logits = transformer_model(features_tensor)
                            probs = F.softmax(logits, dim=1)
                            pred_class = torch.argmax(probs, dim=1).item()
                            confidence = probs[0, pred_class].item()
                        
                        drowning_predictions[track_id].append({
                            'frame_id': frame_count,
                            'pred_class': pred_class,  
                            'confidence': confidence
                        })
                        
                        status = "DROWNING" if pred_class == 0 else "NORMAL"
                        print(f"  Track ID {track_id}: {status} (confidence: {confidence:.2f})")
                        
                        for _ in range(sequence_overlap):
                            if len(track_features[track_id]) > sequence_overlap:
                                track_features[track_id].popleft()
                                track_crops[track_id].popleft()
                    else:
                        frames_needed = sequence_length - len(track_features[track_id])
                        print(f"  Track ID {track_id}: Collecting frames ({frames_needed} more needed)")
        
        gt_boxes = None
        
        output_frame = draw_drowning_tracks(frame.copy(), tracks, drowning_predictions, frame_count, gt_boxes)
      
        frame_name = os.path.basename(frame_file)
        output_path = os.path.join(output_dir, f"tracked_{frame_name}") if output_dir else None
        cv2.imwrite(output_path, output_frame)

        if drowning_output_dir:
            drowning_detected_in_frame = any(
                pred['frame_id'] == frame_count and pred['pred_class'] == 0
                for preds in drowning_predictions.values()
                for pred in preds
        )
        if drowning_detected_in_frame:
            drowning_path = os.path.join(drowning_output_dir, frame_name)
            cv2.imwrite(drowning_path, output_frame)

    
    print("\nProcessing complete.")
    
    
    
    print("\nFinal Drowning Detection Summary:")
    for track_id, preds in drowning_predictions.items():
        drowning_count = sum(1 for p in preds if p['pred_class'] == 0)
        normal_count = len(preds) - drowning_count
        
        if drowning_count > normal_count:
            overall_status = "DROWNING"
        else:
            overall_status = "NORMAL"
            
        print(f"Track ID {track_id}: {overall_status} ({drowning_count}/{len(preds)} drowning predictions)")
    
    return {
        'tracks_history': tracks_history,
        'drowning_predictions': drowning_predictions
    }

def draw_drowning_tracks(frame, tracks, drowning_predictions, current_frame, gt_boxes=None):
    for track in tracks:
        if not track.is_confirmed():
            continue
            
        track_id = track.track_id
        ltrb = track.to_ltrb()
        x1, y1, x2, y2 = map(int, ltrb)
        
        is_drowning = False
        confidence = 0.0
        
        track_preds = drowning_predictions.get(track_id, [])
        if track_preds:
            valid_preds = [p for p in track_preds if p['frame_id'] <= current_frame]
            if valid_preds:
                latest_pred = max(valid_preds, key=lambda p: p['frame_id'])
                is_drowning = latest_pred['pred_class'] == 0  
                confidence = latest_pred['confidence']
        
        if is_drowning:
            color = (0, 0, 255)  
        else:
            color = (0, 255, 0)  
        
        cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
        
        if track_preds:
            status = "DROWNING" if is_drowning else "NORMAL"
            label = f"ID:{track_id} | {status} ({confidence:.2f})"
        else:
            label = f"ID:{track_id} | Processing..."
            
        cv2.putText(frame, label, (x1, y1-10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
    
    if gt_boxes is not None:
        for gt_id, box in gt_boxes.items():
            x1, y1, x2, y2, cls = box
            
            color = (0, 255, 255)  
            
            cv2.rectangle(frame, (x1, y1), (x2, y2), color, 1)
            
            status = "DROWNING" if cls == 1 else "NORMAL"
            label = f"GT:{gt_id} | {status}"
            cv2.putText(frame, label, (x1, y1-10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1)
    
    cv2.putText(frame, f"Frame: {current_frame}", (10, 30), 
               cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
    
    return frame



def calculate_iou(box1, box2):
    if not isinstance(box1, np.ndarray):
        box1 = np.array(box1)
    if not isinstance(box2, np.ndarray):
        box2 = np.array(box2)
        
    x_left = max(box1[0], box2[0])
    y_top = max(box1[1], box2[1])
    x_right = min(box1[2], box2[2])
    y_bottom = min(box1[3], box2[3])
    
    if x_right < x_left or y_bottom < y_top:
        return 0.0
    
    intersection_area = (x_right - x_left) * (y_bottom - y_top)
    
    box1_area = (box1[2] - box1[0]) * (box1[3] - box1[1])
    box2_area = (box2[2] - box2[0]) * (box2[3] - box2[1])
    union_area = box1_area + box2_area - intersection_area
    
    iou = intersection_area / union_area if union_area > 0 else 0.0
    
    return iou

    
    gt_data = {}
    
    with open(gt_file, 'r') as f:
        for line in f:
            parts = line.strip().split(',')
            if len(parts) < 6:
                parts = line.strip().split()
                
            if len(parts) >= 6:
                frame_id = int(parts[0])
                obj_id = int(parts[1])
                x = float(parts[2])
                y = float(parts[3])
                w = float(parts[4])
                h = float(parts[5])
                
                cls = 1  
                if len(parts) >= 8:
                    cls = int(parts[7])
                
                x1, y1 = x, y
                x2, y2 = x + w, y + h
                
                if frame_id not in gt_data:
                    gt_data[frame_id] = {}
                    
                gt_data[frame_id][obj_id] = (int(x1), int(y1), int(x2), int(y2), cls)
    
    return gt_data


