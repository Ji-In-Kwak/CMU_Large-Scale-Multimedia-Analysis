# CMU_11775 Large Scale Multimedia Analysis
## Homework 3

### Early Fusion

In order to implement the early fusion model, you need to change the path of extracted features in .ipynb file. There are audio_data_path and video_data_path. 
```
audio_data_path = [your path of audio feature directory]
video_data_path = [your path of video feature directory]
```
Then, run the ipynb file and it will save the test set prediction values as csv file in 'output' file. 


### Late Fusion

Same as the early fusion, change the path of your audio and video feature directory as follows. 

```
audio_data_path = [your path of audio feature directory]
video_data_path = [your path of video feature directory]
```
By running the following code in .ipynb file, two MLP classifiers are trained respectively and select the final predicted class in the last part of the code. 

