# Rotation Prediction


This repository contains code for training and testing deep learning models for rotation prediction on the COCO2017 dataset. The model receives two images (imageA, imageB) in the input and predicts the rotation angle from imageA to imageB.

---

## Examples
<table>
  <tr>
    <td><img src="examples/val1.png" width="150"><br><em>Pred: 191.00°, Actual: 190.00°</em></td>
    <td><img src="examples/val2.png" width="150"><br><em>Pred: Pred: 114.00°, Actual: 116.00°</em></td>
    <td><img src="examples/val3.png" width="150"><br><em>Pred: Pred: 271.00°, Actual: 272.00°</em></td>
  </tr>
  <tr>
    <td><img src="examples/val4.png" width="150"><br><em>Pred: 274.00°, Actual: 272.00°</em></td>
    <td><img src="examples/val5.png" width="150"><br><em>Pred: Pred:Pred: 256.00°, Actual: 257.00°</em></td>
    <td><img src="examples/val6.png" width="150"><br><em>Pred: Pred: 201.00°, Actual: 203.00°</em></td>
  </tr>
</table>

## Experimental Results

A wandb report with experimental results can be found [here](https://wandb.ai/dtriantafyllidou/siamese-network-rotation-prediction-COLAB/reports/Rotation-Prediction--VmlldzoxMTUyOTYzMQ?accessToken=gggwj17b5s0uvd234q3kmlnobnuk3ycyqbjzgnecyk4690mcivvdykoysuuqkjfj).

## 🔧 Installation

1. Create a new Conda environment:
    ```bash
    conda create --name siameserot python=3.10
    conda activate rotpred
    ```

3. Install the required dependencies:
    ```bash
    pip install -r requirements.txt
    ```

## 📌 Project Structure

Download the COCO2017 data from [here](https://www.kaggle.com/datasets/awsaf49/coco-2017-dataset). Move the dataset in the ./data folder.
The project contains two scripts that can be used for training and inference.

```
.
├── README.md
├── data
│   └── coco2017
│       ├── test2017
│       │   └── 000000000016.jpg
│       ├── test2017_angles.csv
│       ├── test2017_image_pairs
│       │   ├── 000000002884_original.jpg
│       │   └── 000000002884_rotated_188.59.jpg
│       ├── train2017
│       │   └── 000000000009.jpg
│       ├── val2017
│       │   └── 000000000139.jpg
│       └── val2017_angles.csv
├── inference.py
├── requirements.txt
├── train.py
└── utils.py
```


## Model Training
To train a model first install the required packages from the terminal, and then run the following command:

```
python train.py \
--num_epochs=<number_of_epochs> \
--root_dir=<root_directory> \
--save_freq=<save_frequency> \
--print_freq=<print_frequency> \
--batch_size=<batch_size_for_training> \
--load_pretrained_weights=<whether_to_load_pretrained_weights> \
--load_epoch_weights=<epoch_weights_to_load> \
--do_train=<whether_to_train> \
--wandb_log=<whether_to_log_wandb> \
--num_workers=<number_of_workers>
```

The arguments are described below:

- `--num_epochs`:Number of epochs for training. Default value is 20.
- `--root_dir`:  Root directory for the project.
- `--save_freq`: Frequency of saving the model weights. Default value is 1.
- `--print_freq`: Frequency of printing the training progress. Default value is 100.
- `--batch_size`: Batch size for model training. Default value is 32.
- `--load_pretrained_weights`: If True, loads the pre-trained weights. Default value is True.
- `--load_epoch_weights`: Epoch weights to load for pre-trained model. Default value is 1.
- `--do_train`: If True, performs training, otherwise performs evaluation.
- `--wandb_log`: If True, logs the training and evaluation metrics to Weights & Biases (W&B). Default value is True.
- `--num_workers`:  Number of workers for data loading. Default value is 8.

one example coommand for training using the default parameters would be:

```
python train.py --do_train True 
```

## Evaluation

To evaluate model on COCO2017 test set the same script can be used with different parameters:

```
python train.py --load_pretrained_weights True --load_epoch_weights 2
```


## Inference

The following script accepts two images and predicts a rotation angle:
```
python inference.py \
--model_save_path=<model_save_path> \
--filename_1=<filename_1> \
--filename_2=<filename_1> 
```

The arguments are described below,

- `--model_save_path`: Path where the trained model checkpoint is saved.
- `--filename_1`: Path to first image.
- `--filename_2`: Path to second image.

one example coommand for using this script would be,

```
python inference.py \
--model_save_path=./siamese_network_checkpoint_epoch_2.pth \
--filename_1="./data/coco2017/test2017_image_pairs/000000002884_original.jpg" \
--filename_2="./data/coco2017/test2017_image_pairs/000000002884_rotated_188.59.jpg" 
```

## Contact

For any questions please contact danaitri22@gmail.com