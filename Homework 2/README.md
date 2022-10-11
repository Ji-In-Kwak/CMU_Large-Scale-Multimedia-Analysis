# CMU_Large Scale Multimedia Analysis
## Homework 2

### Task 1 : SIFT

Using the extracted SIFT features with 150 dimension, we can train the MLP classifer. Before running the following code, we should change the file name from code/modules/mlp_sift.py to code/modules/mlp.py. 
```
python code/run_mlp.py sift_150 --num_features 150 --learning_rate 0.0001 --train_val_list_file data/labels/train_val.csv --feature_dir data/bow_sift_150 --batch_size 1024
```


### Task 2 : CNN

Extract the feature from Resnet18 by running all train_val.csv and test_for_students.csv
```
python code/run_cnn.py data/labels/xxx.csv
```

Then, train the MLP classifier by running
```
python code/run_mlp.py cnn --num_features 512 --learning_rate 0.0001 --train_val_list_file data/labels/train_val.csv --feature_dir data/cnn
```


### Task 3 : 3D CNN

We can change the model between r3d_18 or r2plus1d_18 with following code in code/run_cnn3d.py. 
```
CNN3DFeature(cnn_resources,
                         # TODO: choose the model, weight, and node to use
                         model_name='r3d_18',
                         weight_name='R3D_18_Weights',
                         node_name='avgpool',
                         replica_per_gpu=self.args.replica_per_gpu)

CNN3DFeature(cnn_resources,
                         # TODO: choose the model, weight, and node to use
                         model_name='r2plus1d_18',
                         weight_name='R2Plus1D_18_Weights',
                         node_name='avgpool',
                         replica_per_gpu=self.args.replica_per_gpu)
```

Extract the feature from 3D CNN models by running all the train_val.csv and test_for_students.csv. 
```
python code/run_cnn.py data/labels/xxx.csv
```

In order to train the features from Resnet3D, run 
```
python code/run_mlp.py cnn3d --num_features 512 --learning_rate 0.0001 --train_val_list_file data/labels/train_val.csv --feature_dir data/cnn3d 
```

In order to train the features from Resnet2D_plus_1D, run
```
python code/run_mlp.py cnn2d_1d --num_features 512 --learning_rate 0.0001 --train_val_list_file data/labels/train_val.csv --feature_dir data/cnn2d_1d 
```


### Evaluation

In order to test the validation set accuracy and confusion matrix, we made run_eval.py file for the evaluation. For CNN and 3D-CNN model, if we know the mlp file path, we can run the evalutaion as follows. [name] represents the folder name under data/mlp folder which is used when training MLP. [version_num] represents the number of version for the experiments and where the checkpoints and test_prediction csv files are included. 
```
python code/run_eval.py [name] [version_num]
```
