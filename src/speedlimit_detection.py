import cv2
import pytesseract
import numpy as np
import matplotlib as plt
from pathlib import Path
import torch

# denormalize an image
def denormalize(im):
    imagenet_stats = np.array([[0.485, 0.456, 0.406], [0.229, 0.224, 0.225]])
    im = im * imagenet_stats[1] + imagenet_stats[0]
    return np.clip(im, 0, 1)

# Crop out an image with a bounding box
def crop(im, r, c, target_r, target_c):
    return im[r:r+target_r, c:c+target_c]

# save the speedlimit signs into a folder 
def save_speedlimit_signs(data_loader, model, output_folder):
    # Set the model to evaluation mode
    model.eval()
    count = 0
    # Loop through the dataset provided by data_loader
    for x, y_class, y_bb in data_loader:
        x = x.float()
        # Run the model to get the predicted bounding box
        out_class, out_bb = model(x)
        # find the predicted class
        _, preds = torch.max(out_class, 1)
        # Iterate over each prediction
        for idx, pred in enumerate(preds):
            # check if the prediction is speedlimit sign
            if pred == 0:
                # Extract bounding box coordinates
                bb = y_bb[idx].detach().numpy().astype(int)
                # Adjust the shape of the image
                image = x[idx].cpu().detach().numpy()
                image = np.rollaxis(image, 0, 3)
                # Denormalize the image
                image = denormalize(image)
                # Crop the image using the bounding box
                cropped_image = crop(image, bb[0], bb[1], bb[2], bb[3])
                # Convert the cropped image to an 8-bit format
                cropped_image = (cropped_image * 255).astype(np.uint8)
                # Define the save path
                save_path = output_folder / f"cropped_{count}.png"
                count += 1
                # Save the cropped image 
                cv2.imwrite(str(save_path), cv2.cvtColor(cropped_image, cv2.COLOR_RGB2BGR))


def detect_speedlimit(image_path):
    # Load Image
    image = cv2.imread(image_path)

    # Checks if the image is successfully loaded
    if image is None or image.size == 0:
        print("Failed to load image")
        return

    # Convert to RGB and Display Cropped Image
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    plt.imshow(image)
    plt.title("Image")
    plt.show()
    # Convert to Grayscale
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # Apply Threshold and Display
    thresh = cv2.threshold(gray, 200, 255, cv2.THRESH_BINARY_INV)[1]
    plt.imshow(thresh)
    plt.title("After Threshold Image")
    plt.show()

    # Extract text using OCR
    text = pytesseract.image_to_string(thresh, config=r'--oem 1 --psm 8 digits')
    cleaned_text = "".join([t for t in text if t.isdigit()]).strip()
    
    return cleaned_text

# Iterate through a folder to get the speedlimit detection
def get_folder_speedlimit(folder_path):
    folder_path = Path(folder_path)
    if not folder_path.exists():
        print("Folder not found")
        return
    # find the prediction on the speedlimit on the image
    for image_file in folder_path.iterdir():
          detect_text = detect_speedlimit(image_file)
          print(detect_text)