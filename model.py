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

"""

    ### Training flow:
    * save first image for validation
    * after that use input: validation_save_chance
        * randomly save images to the validation folder

        $datasets_dir
            /1
                /val
                    /images
                    /labels
                /train
                    /images
                    /labels

                dataset.yaml
                full-dataset.yaml
                12343244.png
                12342134.txt

    * on train end:
        move image to train/images
        move annotation to train/labels

    * models:

        $weights_dir
            /1
                ?

        * naming scheme:
            [original_name]

"""


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
            print(trainer)

        self.model.add_callback("on_train_end", on_train_end_cb)

        #
        self.storage_client = Minio(S3_ENDPOINT, access_key=S3_ACCESS_KEY, secret_key=S3_SECRET_KEY)

        self.from_name, self.to_name, self.value, self.labels = get_single_tag_keys(
            self.parsed_label_config, 'PolygonLabels', 'Image'
        )

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
        print(data)
        if event == 'ANNOTATION_CREATED':
            pass

        self.execute_single_image_train(data)

    def execute_single_image_train(self, data):
        taskId = data['task']['id']
        print(f"Executing training run for task {taskId} on project [{data['project']['id']}]{data['project']['title']}")

        file_name = self.save_training_image(taskId, urlparse(data['task']['data'][self.value]))
        self.save_training_labels(taskId, data['annotation'], file_name)
        self.generate_classes_txt(taskId)
        self.generate_dataset_yaml(taskId)

        thread = threading.Thread(target=train_model, kwargs={'projectId': self.projectId, 'taskId': taskId})
        thread.start()
        print("Training started on new thread")

    def save_training_image(self, taskId, s3url):
        print("Saving task image:", s3url.path)
        response = self.storage_client.get_object(s3url.netloc, s3url.path)
        image = Image.open(response)
        images_dir = os.path.join(self.get_project_datasets_dir(), str(taskId), 'images')
        if not os.path.exists(os.path.join(images_dir, 'train')):
            os.makedirs(os.path.join(images_dir, 'train'))
        if not os.path.exists(os.path.join(images_dir, 'val')):
            os.makedirs(os.path.join(images_dir, 'val'))

        file = os.path.basename(s3url.path)
        image.save(os.path.join(images_dir, 'train', file), 'PNG')

        root, ext = os.path.splitext(file)

        return root

    def save_polygon_labels_from_result(self, annotation_result, path):
        print("Processing annotation data")
        segm = []
        for result in annotation_result:
            points = np.array(result['value']['points']) / \
                np.array([result['original_width'], result['original_height']])

            points_list = points.flatten().tolist()

            points_str = ' '.join(str(point) for point in points_list)
            for label in result['value']['polygonlabels']:
                segm.append(f"{self.get_label_id(label)} {points_str}")

        if not os.path.exists(path):
            os.makedirs(path)

        with open(os.path.join(labels_dir, 'train', file_name + '.txt'), 'w') as f:
            f.write('\n'.join(segm))

    def generate_dataset_yaml(self, taskId):
        print("Generating dataset.yaml")
        labels = {}

        for index, label in enumerate(self.parsed_label_config['label']['labels']):
            labels[index] = label

        dataset = {
            'path': os.path.join(self.get_project_datasets_dir(), str(taskId)),
            'train': 'images/train',
            'val': 'images/val',
            'names': labels
        }

        with open(os.path.join(self.get_project_datasets_dir(), str(taskId), 'dataset.yaml'), 'w') as f:
            f.write(yaml.dump(dataset))

    def generate_classes_txt(self, taskId):
        print("Generated classes.txt")
        dataset_dir = os.path.join(self.get_project_datasets_dir(), str(taskId))
        with open(os.path.join(dataset_dir, 'classes.txt'), 'w') as f:
            f.write('\n'.join(self.parsed_label_config['label']['labels']))

    def get_project_datasets_dir(self):
        return os.path.join(os.path.abspath(settings['datasets_dir']), str(self.projectId))

    def get_project_runs_dir(self):
        return os.path.join(os.path.abspath(settings['runs_dir']), str(self.projectId))

    def get_project_weights_dir(self):
        return os.path.join(os.path.abspath(settings['weights_dir']), str(self.projectId))


def train_model(projectId, taskId):
    backend = YOLOv8Model(projectId)

    backend.model.train(
        data=os.path.join(backend.get_project_datasets_dir(), str(taskId), 'dataset.yaml'),
        epochs=10,
        project=backend.get_project_runs_dir(),
        name=f"{str(taskId)}-single",
        mask_ratio=1,
        overlap_mask=True,
        val=False
    )


def get_label_id(parsed_label_config, label):
    return parsed_label_config['label']['labels'].index(label)
