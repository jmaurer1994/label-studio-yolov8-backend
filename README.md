# Quickstart

1. Build and start Machine Learning backend on `http://localhost:9090`

```bash
docker-compose up
```

or

```bash
label-studio-ml start .
```


2. Validate that backend is running

```bash
$ curl http://localhost:9090/health
{"status":"UP"}
```

3. Connect to the backend from Label Studio: go to your project
   `Settings -> Machine Learning -> Add Model` and specify `http://localhost:9090` as a URL.

    * This will download the yolov8n-seg.pt model via the ultralytics module
    * Run full_train.py to train the current model against the full dataset
    * On new annotations, the annotation will be saved for validation or used for training
        * If used for training, the current model will be trained for 10 epochs and then updated
          as the active version for the backend




    * save first image for validation
    * after that, every 10th


## Directory structure

Base directory are configured via ultralytics settings

### Datasets

```
$datasets_dir
    /$projectId
        /val
            /images
            /labels
        /train
            /images
            /labels
        /current
            /images
            /labels
    dataset.yaml
    dataset-full.yaml
```

### Models

```
$weights_dir
    /$projectId
        
```

### Training Runs

```
$runs_dir
    /$projectId
        /$taskId
            /single <- single train experiment name

        /full <- full train experiment name

```

## fit() flow
    
* on new annotation:
    * save for val or train
    * trigger yolov8 training on new thread



    * on train end:
        * if single:
            * move image to train/images
            * move annotation to train/labels
        * update model version
            * On single increment last grouping
            * On full train append training set image count and reset single counter to 0

