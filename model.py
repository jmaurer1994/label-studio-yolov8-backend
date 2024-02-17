from typing import List, Dict, Optional
from label_studio_ml.model import LabelStudioMLBase
from label_studio_ml.utils import get_single_tag_keys, get_choice, is_skipped
import numpy as np
from ultralytics import YOLO
from ultralytics import settings
from PIL import Image
import os
from dotenv import load_dotenv
from minio import Minio
from urllib.parse import urlparse
import threading
import yaml

load_dotenv()

LABEL_STUDIO_URL = os.getenv("LABEL_STUDIO_URL", "")
LABEL_STUDIO_API_TOKEN = os.getenv("LABEL_STUDIO_API_TOKEN", "")

# Using bucket storage - self hosted minio was easy
S3_ENDPOINT = os.getenv("S3_ENDPOINT", "")
S3_ACCESS_KEY = os.getenv("S3_ACCESS_KEY", "")
S3_SECRET_KEY = os.getenv("S3_SECRET_KEY", "")


class YOLOv8Model(LabelStudioMLBase):
    def __init__(self, projectId, **kwargs):
        # Call base class constructor
        super(YOLOv8Model, self).__init__(**kwargs)
        # Initialize your model here

        self.projectId = projectId
        if self.get('model_version') == 'INITIAL' or not self.get('model_version'):
            self.set('model_version', 'yolov8n-seg')

        self.model = YOLO(os.path.join(self.get_project_weights_dir(), self.get('model_version') + '.pt'))

        def on_train_end_cb(trainer):
            print("Training finished")
            print(trainer.best)
            self.move_new_weights(trainer.best, trainer.args.name)

        self.model.add_callback("on_train_end", on_train_end_cb)

        self.storage_client = Minio(S3_ENDPOINT, access_key=S3_ACCESS_KEY, secret_key=S3_SECRET_KEY)
        try:
            self.from_name, self.to_name, self.value, self.labels = get_single_tag_keys(
                self.parsed_label_config, 'PolygonLabels', 'Image'
            )
        except TypeError:
            pass

    def predict(self, tasks: List[Dict], context: Optional[Dict] = None, **kwargs) -> List[Dict]:
        # images will only have one task for now (segmatation - polyline
        task = tasks[0]
        parsed_image_url = urlparse(task['data'][self.value])
        response = self.storage_client.get_object(parsed_image_url.netloc, parsed_image_url.path)
        # Get URL and loading image
        image = Image.open(response)

        # Height and width of image
        original_width, original_height = image.size

        # Create list for predictions and variable for scores
        predictions = []
        score = 0
        i = 0

        # Get prediction using model
        results = self.model.predict(image)

        # Getting mask segments, boxes from model prediction
        for result in results:
            if result.masks is None:
                continue
            for i, (box, segm) in enumerate(zip(result.boxes, result.masks.xy)):

                # 2D array with polygon points
                polygon_points = segm / \
                    np.array([original_width, original_height]) * 100

                polygon_points = polygon_points.tolist()

                # Adding dict to prediction
                predictions.append({
                    "from_name": self.from_name,
                    "to_name": self.to_name,
                    "id": str(i),
                    "type": "polygonlabels",
                    "score": box.conf.item(),
                    "original_width": original_width,
                    "original_height": original_height,
                    "image_rotation": 0,
                    "value": {
                        "points": polygon_points,
                        "polygonlabels": [self.labels[int(box.cls.item())]]  # I believe this is indexed by the order classes text, which should match the other of the label config
                    }})

                # Calculating score
                score += box.conf.item()

        print(f"Prediction Score is {score:.3f}.")

        # Dict with final dicts with predictions
        final_prediction = [{
            "result": predictions,
            "score": score / (i + 1),
            "model_version": self.get('model_version')
        }]

        return final_prediction

    def fit(self, event, data, **kwargs):
        """
        This method is called each time an annotation is created or updated

        """
        print(self.labels)
        print(data)
        if event == 'ANNOTATION_CREATED':
            self.save_annotation(data)
        elif event == 'ANNOTATION_UPDATED':
            self.save_annotation(data)

    def save_annotation(self, data):
        taskId = data['task']['id']
        print(f"Executing training run for task {taskId} on project [{data['project']['id']}]{data['project']['title']}")

        # decide if image will be used for validation or training
        training_required = False
        training_images = get_image_count(os.path.join(self.get_project_dataset_dir(), 'train', 'images'))
        validation_images = get_image_count(os.path.join(self.get_project_dataset_dir(), 'val', 'images'))
        total_images = training_images + validation_images
        if validation_images == 0 or total_images % 10 == 0:
            path = os.path.join(self.get_project_dataset_dir(), 'val')
        else:
            path = os.path.join(self.get_project_dataset_dir(), 'current')
            training_required = True

        s3url = urlparse(data['task']['data'][self.value])
        print("Saving task image:", s3url.path)
        response = self.storage_client.get_object(s3url.netloc, s3url.path)
        image = Image.open(response)

        file = os.path.basename(s3url.path)
        image.save(os.path.join(os.path.join(path, 'images'), file), 'PNG')
        file_name, ext = os.path.splitext(file)

        print("Saving annotation")
        save_to_path(self.polygon_results_tostring(data['annotation']), os.path.join(path, 'labels'), file_name + '.txt')

        print("Saving classes.txt")
        save_to_path('\n'.join(self.parsed_label_config['label']['labels']), self.get_project_dataset_dir(), 'classes.txt')

        print("Generating dataset.yaml")
        save_to_path(self.generate_dataset_yaml(), self.get_project_dataset_dir(), 'dataset.yaml')
        save_to_path(self.generate_full_dataset_yaml(), self.get_project_dataset_dir(), 'dataset-full.yaml')
        if training_required:
            thread = threading.Thread(target=train_model, kwargs={'projectId': self.projectId, 'taskId': taskId})
            thread.start()
            print("Training started on new thread")

    def polygon_results_tostring(self, annotation_data):
        segm = []

        for result in annotation_data['result']:
            points = np.array(result['value']['points']) / \
                np.array([result['original_width'], result['original_height']])

            points_list = points.flatten().tolist()

            points_str = ' '.join(str(point) for point in points_list)
            for label in result['value']['polygonlabels']:
                segm.append(f"{self.get_label_id(label)} {points_str}")

        return '\n'.join(segm)

    def generate_dataset_yaml(self):
        labels = {}

        for index, label in enumerate(self.parsed_label_config['label']['labels']):
            labels[index] = label

        dataset = {
            'path': os.path.join(self.get_project_dataset_dir()),
            'train': 'current/images',
            'val': 'val/images',
            'names': labels
        }

        return yaml.dump(dataset)

    def generate_full_dataset_yaml(self):
        labels = {}

        for index, label in enumerate(self.parsed_label_config['label']['labels']):
            labels[index] = label

        dataset = {
            'path': os.path.join(self.get_project_dataset_dir()),
            'train': 'train/images',
            'val': 'val/images',
            'names': labels
        }

        return yaml.dump(dataset)

    def get_project_dataset_dir(self):
        return os.path.join(os.path.abspath(settings['datasets_dir']), str(self.projectId))

    def get_project_runs_dir(self):
        return os.path.join(os.path.abspath(settings['runs_dir']), str(self.projectId))

    def get_project_weights_dir(self):
        return os.path.join(os.path.abspath(settings['weights_dir']), str(self.projectId))

    def get_label_id(self, label):
        return self.parsed_label_config['label']['labels'].index(label)

    def move_new_weights(self, bestPath, experimentName):
        currentVersion = self.get('model_version')

        vl = currentVersion.split('.')

        if experimentName.startswith("single"):
            if len(vl) == 1:
                vl.append(str(1))
            else:
                vl[len(vl) - 1] = str(int(vl[len(vl) - 1]) + 1)  # increment last by 1
        else:
            tp = os.path.join(self.get_project_dataset_dir(), 'train', 'images')
            vl.append(str(get_image_count(tp)))
            vl.append(str(0))

        newVersion = '.'.join(vl)
        self.set('model_version', newVersion)
        os.rename(bestPath, os.path.join(self.get_project_weights_dir(), newVersion + '.pt'))
        print(f"Model updated to version {newVersion}")


def train_model(projectId, taskId):
    backend = YOLOv8Model(projectId)

    backend.model.train(
        data=os.path.join(backend.get_project_dataset_dir(), 'dataset.yaml'),
        epochs=10,
        project=backend.get_project_runs_dir(),
        name=f"{str(taskId)}/single",
        mask_ratio=1,
        overlap_mask=True)


def train_model_full(projectId, epochs):
    backend = YOLOv8Model(projectId)

    backend.model.train(
        data=os.path.join(backend.get_project_dataset_dir(), 'dataset-full.yaml'),
        epochs=epochs,
        project=backend.get_project_runs_dir(),
        name='full',
        mask_ratio=1,
        overlap_mask=True)


def save_to_path(data, path, file):
    if not os.path.exists(path):
        os.makedirs(path)

    with open(os.path.join(path, file), 'w') as f:
        f.write(data)


def get_image_count(path):
    # image_extensions = ['.jpg', '.jpeg', '.png', '.gif', '.bmp', '.tiff']
    image_count = 0

    for root, dirs, files in os.walk(path):
        for file in files:
            # if any(file.lower().endswith(ext) for ext in image_extensions):
            if file.lower().endswith('.png'):
                image_count += 1

    return image_count
