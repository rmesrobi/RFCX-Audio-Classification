# Rainforest Connection Species Audio Detection

<p align="center">
    <img src="images/header.png" width='800'/>
</p>


## Overview

The [Rainforest Connection Species Audio Detection](https://www.kaggle.com/c/rfcx-species-audio-detection) competition is currently live on Kaggle. The goal of the competition is to detect a variety of bird and frog species in a tropic soundscape recording. The competition database contains a series of acoustically complex, one-minute recordings containing at least one call of a known wildlife species. The database also includes true positive and false positive csv files that can be used to help train the detection model.

## EDA & Audio Processing

The dataset includes over 4,700 one-minute audio recordings that are designated for model training. Data from these recordings can be found in the train_tp (1,132 rows) and train_fp (3,958 rows) csv files. The csv files include detail about the recordings such as: recording_id, species_id, songtype_id, start and end time, and min and max frequencies. The test audio database is comprised of nearly 2,000 one-minute audio recordings. No labels are included for these recordings.

My analysis and model are based on only the true positive recordings and csv file. After cleaning the data I had 1,088 recordings to work with. The graphs below summarize the metadata for these recordings.

<p align="center">
    <img src="images/count_of_species_in_dataset.png" width='600'/>
</p>

<p align="center">
    <img src="images/bandwidth_by_species_id.png" width='600'/>
    <img src="images/call_duration_by_species_id.png" width='600'/>
</p>

I used librosa, a Python module for audio and music processing to extract features from the audio recordings. For this project, I focused on analyzing Mel Spectrograms, which will be explained later. For now, we will start with the concept of sound.

Sound is basically a sequence of vibrations in varying pressure strengths. Loosely speaking, visualizing sound really means visualizing airwaves. A two-dimensional representation of a song can be expressed in a waveplot, which illustrates amplitude over time. Here are a few examples of waveplots from the dataset. The highlighted sections of the plots illustrate the moment in time the species is heard in the audio clip. The last waveplot is an example of an audio recording with five distinct calls from four different species of animals.

<p align="center">
    <img src="images/waveplot_10.png" width='400'/>
    <img src="images/waveplot_12.png" width='400'/>
    <img src="images/waveplot_14.png" width='400'/>
    <img src="images/waveplot_23.png" width='400'/>
</p>

<p align="center">
    <img src="images/waveplot_multi-class.png" width='800'/>
</p>

Skipping ahead a few steps, we wind up with a spectrogram. A spectrogram is a visual representation of the spectrum of frequencies of a signal as it varies over time- showing which frequencies are active at a particular moment. A spectrogram transforms frequency to a particular scale and amplitude to decibels. Further complicating things, the Mel Scale is the result of some non-linear transformation of the frequency scale. This Mel Scale is constructed such that sounds are of equal distance from each other on the Mel Scale. This is how humans hear "sound".

<p align="center">
    <img src="images/spec_example.png" width='250'/>
    <img src="images/spec_log_example.png" width='250'/>
    <img src="images/mel_spec_example.png" width='250'/>
</p>

Spectrograms are great for computers to train on. Here are a few examples of Mel Spectrograms from the dataset:

<p align="center">
    <img src="images/spec_23.png" width='200'/>
    <img src="images/spec_14.png" width='200'/>
    <img src="images/spec_12.png" width='200'/>
    <img src="images/spec_10.png" width='200'/>
</p>

### Recap:
The Mel Spectrogram is the result of the following pipeline:

Separate audio into windows > Compute the Fast Fourier Transform (FFT) > Generate Mel Scale > Generate Spectrogram


<p align="center">
    <img src="images/hipster_joe.png" width='200'/>
</p>








## Modeling


<p align="center">
    <img src="images/tpfp_example.png" width='600'/>
</p>

A detection model will need to identify true positive and false positive predictions from the audio clips. The spectrogram above is an example of an audio clip with multiple true and false positives.

### Image Processing
1. Created clips of all true positive audio samples, which I then used to generate Mel Spec images.
2. Saved image files into a directory and sub-directory
3. Used TensorFlow to get labeled Mel Spec image data, labeled by species_id, as a generator object
4. Adjusted image size to (180, 180, 3)
5. Normalized data to a 0 - 1 scale for the CNN model using TensorFlow preprocessing

The first model I used consisted of three convultion blocks with a max pool layer in each of them. The model is activated by a relu activation function at each layer and uses a softmax activation function for multi-class classification. When compiling the model, I used the Adam optimizer and SparseCategoricalCrossentropy loss function. I passed the metrics argument to be able to view training and validation accuracy for each training epoch.

<p align="center">
    <img src="images/model_summary.png" width='600'/>
</p>

To reduce complexity, I trained the model using only true positive recordings. I reduced the audio clip dimensions so that only the true positive occured in the clip. Here's the function and images I used to create images for the model:


```
def create_spec_train_val(filename):
  plt.interactive(False)
  fig, ax = plt.subplots(figsize = (3,3))
  ax.get_xaxis().set_visible(False)
  ax.get_yaxis().set_visible(False)
  ax.set_frame_on(False)
  species_id, recording_id, name, t_min, t_max = get_data(filename)
  data, sr = read_flac_file(recording_id)
  #creates clip of data between t_min and t_max
  clip = data[int(t_min * sr): int(t_max * sr)]
  spec = librosa.feature.melspectrogram(clip, sr = sr)
  spec = librosa.power_to_db(spec, ref = np.max)
  librosa.display.specshow(spec, sr = sr, x_axis = 'time', y_axis = 'mel')
  path = 'train_val_images/'
  save_name = str(name) + '.jpg'
  plt.savefig(path + str(species_id) + '/' + save_name, dpi = 400, bbox_inches = 'tight', pad_inches = 0)
  plt.close()
  fig.clf()
  plt.close(fig)
  plt.close('all')
  del path, save_name, filename, data, fig, ax, spec
  ```

<p align="center">
    <img src="images/spec_grid_2.png" width='400'/>
</p>

### Training and Validation Scores:

<p align="center">
    <img src="images/model_fit.png" width='600'/>
</p>

<p align="center">
    <img src="images/model_train_val.png" width='600'/>
</p>

### Predicting on test data:

<p align="center">
    <img src="images/model_test.png" width='600'/>
</p>

### Confusion Matrix:

<p align="center">
    <img src="images/confusion_matrix.png" width='600'/>
</p>

The next model I tried incorporated data augmentation to generate additional training data. I did this by using experimental Keras Preprocessing Layers. The image augmentation reduced the model's accuracy on both training and validation and also on testing.

```
data_augmentation = keras.Sequential(
    [
     layers.experimental.preprocessing.RandomFlip('horizontal',
                                                  input_shape = (img_height, img_width, 3)),
     layers.experimental.preprocessing.RandomRotation(0.1),
     layers.experimental.preprocessing.RandomZoom(0.1),
    ])
```

Here are a few examples of the augmented images:

<p align="center">
    <img src="images/data_augmentation.png" width='600'/>
</p>

I also included dropout into this model as a form of regularization. However, this model did not perform as well as the first.


### Training and Validation Scores:
<p align="center">
    <img src="images/dropout_train_val.png" width='600'/>
</p>
<p align="center">
    <img src="images/dropout_test.png" width='600'/>
</p>

## Next Steps

I would like to improve the accuracy of the CNN model by adjusting the audio clip size. This would increase the amount of images I could use to train my model. 

I would also like to use other audio features such as mel-frequency cepstral coefficients (MFCCs), implement a Random Forest model, incorporate false positives into training and testing, improve the quality of the mel spectrogram images, and increase the training clip length to the full audio file length.

[Rainforest Audio Detection Data](https://www.kaggle.com/c/rfcx-species-audio-detection/data)
