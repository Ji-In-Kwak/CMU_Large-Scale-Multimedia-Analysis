# CMU_Large Scale Multimedia Analysis
## Homework 1 

### Task 1
Follow the instruction from the Homework 1 github.

### Task 2
After extracting the mfcc features, we can train each of SVM and MLP classifiers.
```
$ python train_svm_multiclass.py bof/ 50 labels/train_val.csv weights/mfcc-50.svm.model
```
```
$ python train_mlp.py bof/ 50 labels/train_val.csv weights/mfcc-50.mlp.model
```
Then, make a prediction of classes for the test dataset.
```
$ python test_svm_multiclass.py weights/mfcc-50.svm.model bof/ 50 labels/test_for_students.csv mfcc-50.svm.csv
```
```
$ python test_mlp.py weights/mfcc-50.mlp.model bof/ 50 labels/test_for_students.csv mfcc-50.mlp.csv
```
The results are saved as csv files which are mfcc-50.svm.csv and mfcc-50.mlp.csv


### Task 3
Convert the video files into mp3 format audio files.
```
$ for file in videos/*;do filename=$(basename $file .mp4);  ffmpeg -y -i $file -ac 1 -ar 22050 -f mp3 mp3/${filename}.mp3; done
```
Then extract the single-vector features from pretrained SoundNet model.
```
python scripts/extract_soundnet_feats.py --feat_layer conv7
```

After extracting the features, then train the MLP classifier as we did in Task 2.
```
$ python train_mlp_v2.py snf/ 1024 labels/train_val.csv weights/snf.mlp.model
```
```
$ python test_mlp_v2.py weights/snf.mlp.model snf/ 1024 labels/test_for_students.csv snf.mlp.csv
```


### Task 4
Install the pretrained PaSST following the 'https://github.com/kkoutini/passt_hear21'.
From Data Preprocessing.ipynb, make a pickle file for train, validation, and test mp3 dataset.
Then train MLP models using mlp_train.ipynb

