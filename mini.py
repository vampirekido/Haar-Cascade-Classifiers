import cv2
import os
from google.colab.patches import cv2_imshow  # Import for displaying images in Colab

# Load the Haar cascade files
face_cascade = cv2.CascadeClassifier('/content/haarcascade_frontalface_default.xml')
eye_cascade = cv2.CascadeClassifier('/content/haarcascade_eye.xml')
smile_cascade = cv2.CascadeClassifier('/content/haarcascade_smile.xml')
eyeglasses_cascade = cv2.CascadeClassifier('/content/haarcascade_eye_tree_eyeglasses.xml')
plate_cascade = cv2.CascadeClassifier('/content/haarcascade_russian_plate_number.xml')

# Function to perform detection
def detect_objects(image_path, cascade, object_name):
    img = cv2.imread(image_path)
    if img is None:
        print(f"Error: Image at {image_path} could not be loaded.")
        return
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    objects = cascade.detectMultiScale(gray, 1.3, 5)

    for (x, y, w, h) in objects:
        cv2.rectangle(img, (x, y), (x+w, y+h), (255, 0, 0), 2)
        cv2.putText(img, object_name, (x, y-10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (36, 255, 12), 2)

    cv2_imshow(img)  # Use cv2_imshow() instead of cv2.imshow()
    cv2.waitKey(0)
    cv2.destroyAllWindows()

# Path to the images directory
images_dir = '/content/images'

# Verify the directory exists and list files
if os.path.exists(images_dir):
    print(f"Directory {images_dir} exists.")
    print("Files in the directory:")
    print(os.listdir(images_dir))
else:
    print(f"Directory {images_dir} does not exist.")

# Loop through the images and apply detection
for filename in os.listdir(images_dir):
    if filename.endswith('.jpg') or filename.endswith('.png'):
        image_path = os.path.join(images_dir, filename)
        print(f"Processing {image_path}")
        detect_objects(image_path, face_cascade, 'Face')
        detect_objects(image_path, eye_cascade, 'Eye')
        detect_objects(image_path, smile_cascade, 'Smile')
        detect_objects(image_path, eyeglasses_cascade, 'Eyeglasses')
        detect_objects(image_path, plate_cascade, 'Number Plate')
