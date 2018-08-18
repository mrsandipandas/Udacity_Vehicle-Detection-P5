# **Vehicle Detection Project**


---

The main objectives of the project are:
* Perform a Histogram of Oriented Gradients (HOG) feature extraction on a labeled training set of images and train a classifier Linear SVM classifier
* Apply a color transform and append binned color features, as well as histograms of color, to the HOG feature vector. 
* Normalize the features and randomize a selection for training and testing.
* Implement a sliding-window technique and use your trained classifier to search for vehicles in images.
* Run the pipeline on a video stream (starting with the test_video.mp4 and later implementing on full project_video.mp4) and create a heat map of recurring detections frame by frame to reject outliers and follow detected vehicles.
* Estimate a bounding box for vehicles detected.

[//]: # (Image References)
[image1]: ./images/car_not_car.png
[image2]: ./images/car_colour_spaces.png
[image3]: ./images/HOG_example.png
[image4]: ./images/sliding_window.png
[image5]: ./images/sliding_window_search.png
[image6]: ./images/bboxes_and_heat.png
[image7]: ./images/labels_map.png
[image8]: ./images/output_bboxes.png
[video1]: ./videos/project_video_out.mp4


### Preprocessing
For a more robust detector, I downloaded the [Udacity labelled dataset](http://bit.ly/udacity-annotations-autti) (annotated by Autti) and pre-prossed the data to have more labelled dataset for the vehicle and non-vehicle classes and converted them to png format. I also down-scaled the images in accordance with the other training dataset provided in the exercise.

```
def loaddata (csvfile):
    vehicle_class = ['car', 'truck']
    with open(csvfile) as inputFile:
        reader = csv.reader(inputFile, delimiter=' ')
        for line in reader:  
            img = cv2.imread('data/'+line[0])    
            x1 = line[1]
            y1 = line[2]
            x2 = line[3]
            y2 = line[4]
            cropped_img = img[int(y1):int(y2), int(x1):int(x2)]
            scaled_img = cv2.resize(cropped_img, (64, 64), interpolation=cv2.INTER_NEAREST)
            if (line[6] in vehicle_class):
		cv2.imwrite('train_data/vehicle/'+line[0].split(".")[0]+'.png', img, [int(cv2.IMWRITE_PNG_COMPRESSION), 4])
            else:
                cv2.imwrite('train_data/novehicle/'+line[0].split(".")[0]+'.png', img, [int(cv2.IMWRITE_PNG_COMPRESSION), 4])
``` 
So, my training data consisted of 21645 vehicle class data and 15488 non-vehicle class data set.

### Histogram of Oriented Gradients (HOG)

#### 1. Explain how (and identify where in your code) you extracted HOG features from the training images.

The code for this step is contained in Step 4: Histogram of Oriented Gradients (HOG) of the IPython notebook.

I started by reading in all the `vehicle` and `non-vehicle` images.  Here is an example of one of each of the `vehicle` and `non-vehicle` classes:

![Vehicle and non-vehicle sample images][image1]

I then explored different color spaces and different `skimage.hog()` parameters (`orientations`, `pixels_per_cell`, and `cells_per_block`).  I grabbed random images from each of the two classes and displayed them to get a feel for what the `skimage.hog()` output looks like.

I explored the different colour spaces and chose YCrCv as this would give optimal gradients.

![Colour spaces in sample images][image2]

Here is an example using the HOG parameters of `orientations=9`, `pixels_per_cell=(8, 8)` and `cells_per_block=(2, 2)`:

![HOG parameters for sample images][image3]

#### 2. Explain how you settled on your final choice of HOG parameters.

I tried various combinations of parameters and settled down with the following parameters.

```
colorspace = 'YCrCb'
orientations=9
pix_per_cell = 16
cells_per_block=(2, 2)
hog_channel = 'ALL' 
spatial_size = (16, 16)
hist_bins = 16

```

With this configuration, I achieve an accuracy of `0.9828` and it requires `0.0014 seconds` to predict 10 labels with classifier. I also serialized the classifier in a pickled file, so that it could be used later on. The code is available under Step 8: Training a classifier.

#### 3. Describe how (and identify where in your code) you trained a classifier using your selected HOG features (and color features if you used them).

I trained a linear SVM using a combination of histogram count, spatial binning and HOG features. The helper function can be found under 
`extract_features` in Step 4. 

### Sliding Window Search

#### 1. Describe how (and identify where in your code) you implemented a sliding window search.  How did you decide what scales to search and how much to overlap windows?

First, I used the implemented a version of `sliding window` function to draw the boxes in half of the image. This was to verify where the search regions would be. The results can be seen below.

![Sliding window][image4]

#### 2. Show some examples of test images to demonstrate how your pipeline is working.  What did you do to optimize the performance of your classifier?

Ultimately I searched using YCrCb, 3-channel HOG features plus and histograms of color in the feature vector, which provided a nice result.  Here are some example of the sliding window search with these features.

![Sliding window search][image5]

---

### Video Implementation

#### 1. Provide a link to your final video output.  Your pipeline should perform reasonably well on the entire project video (somewhat wobbly or unstable bounding boxes are ok as long as you are identifying the vehicles most of the time with minimal false positives.)
Here's a [link to my video result](videos/project_video_out.mp4)


#### 2. Describe how (and identify where in your code) you implemented some kind of filter for false positives and some method for combining overlapping bounding boxes.

I recorded the positions of positive detections in each frame of the video.  From the positive detections I created a heatmap and then thresholded that map to identify vehicle positions.  I then used `scipy.ndimage.measurements.label()` to identify individual blobs in the heatmap.  I then assumed each blob corresponded to a vehicle.  I constructed bounding boxes to cover the area of each blob detected.

Here's an example result showing the heatmap from a series of frames of video, the result of `scipy.ndimage.measurements.label()` and the bounding boxes then overlaid on the last frame of video:

### Here is a image with bounding boxes for the detected vehicles and its corresponding corresponding heatmap:

![Bounding box and heat map][image6]

### Here is the output of `scipy.ndimage.measurements.label()` on the integrated heatmap:
![Labels of the heatmap][image7]

### Here the resulting bounding boxes are drawn onto the video frames in the series:
![Integrated bounding boxes][image8]


---

### Discussion

#### 1. Briefly discuss any problems / issues you faced in your implementation of this project.  Where will your pipeline likely fail?  What could you do to make it more robust?

In the search window sometimes, there are more than one road regions classified as car and as a result this reflects back in thevideo output. I have tried to mitigate this problem by increasing the no. of consecutive frames positive readout. This helps to some extent in the cancelling out of the false positives. However, to absolutely minimize this problem more training data could be collected. Given some more time, I would put the road line detector together as well. Also I would implement a more robust version of sliding window search by controlling the X-direction of search as well. 

