# ANPR System

## Detection using Method 1
In this method, we will employ classical DIP algorithms such as Canny
edge detection contour finding using OpenCV library to extract the
number plate and EasyOCR in order to extract the text from the plate .

The steps involved in this method are listed below :
1. Import and install dependencies
2. Read in image, convert to grayscale and denoise the image
3. Apply dilation filter and find edges for localization
4. Find contours and apply the mask to extract plate
5. Use Easy OCR to read text
6. Render Result

## Detection using Method 2
In this method, we are going to use CNN based method to obtain the
region occupied by the number plate and then use character recognition
to display the text out there We have used a pre trained YOLOv 4 model
to detect number plate in an image.

The following steps are used to implement this method :
1. Import and install dependencies
2. Collecting car images with license plate data
3. Training a CNN model
4. Detecting License Plates
5. Applying OCR to text
6. Output Results

## Comparison of the Two Methods
+ Accuracy : On 50 test images we are getting around 40% accuracy using edge
detection and contours finding while around 85% accuracy in 2 nd method that
leverages neural network to extract number plate.
+ Data Required : No data collection is required in classical DIP method but huge training
data(images and coordinates of number plate in each respective image) is required to
train the multi layer YOLOv4 deep neural network.
+ FPS : Classical method can be used with high FPS to detect number plate and output in
a video based input as compared to YOLOv4 based plate detection.

