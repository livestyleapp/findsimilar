import os
import requests
import shutil
from flask import Flask, jsonify, request
from PIL import Image
from DeepImageSearch import Load_Data, Search_Setup
import threading

app = Flask(__name__)

# Function to download files using requests
def download_file(url, dest):
    response = requests.get(url, stream=True)
    os.makedirs(os.path.dirname(dest), exist_ok=True)
    with open(dest, 'wb') as file:
        for chunk in response.iter_content(chunk_size=8192):
            if chunk:
                file.write(chunk)

# Ensure model weights are downloaded only if they don't already exist
weights_folder = "metadata-files/vgg19"
weights_files = [
    os.path.join(weights_folder, "image_features_vectors.idx"),
    os.path.join(weights_folder, "image_data_features.pkl")
]

if not all(os.path.exists(file) for file in weights_files):
    print("Downloading model weights")
    download_file(
        "https://download1326.mediafire.com/1ga0rvjg6qkgha8oOPFVHY2V_WLYwde6gCWBXT1FQlhATeqS-GNBaZrzcuNuwZSCLs76qScw1vWYEhFSLNiKLjkMJX3Gog3gCSUz2VzV2IRE5ZfZOySSbF5HHww8AU9_dYJNZIriOyGNI4hTosvysfJZvpOY93jRC2z77H9N_66D/ekg10hcq141v5hi/image_features_vectors.idx",
        weights_files[0]
    )
    download_file(
        "https://download1584.mediafire.com/29g7k41bwpqg5eCW0sOQnrNX853Xxi74F66GoMquaXqTH4jeAPyMKr4H8N2vrtBfjscyLDHOJxyMusdAfZEMRq7ZbKzG3hKPPbjf_YIDPcgAf3Keca8UzD6kAPro24YbwhV_63uLYfYWAG9ka8c-L_rCFZ09alLyplZ3OTS_uKfC/0aemngb1qh3t9cn/image_data_features.pkl",
        weights_files[1]
    )
    print("Weights downloaded")
else:
    print("Weights already exist. Skipping download.")

# Load data and set up the model
print("Loading Data")
image_list = Load_Data().from_folder(folder_list=["Data"])
print("Data Loaded")
print("Loading Model")
st = Search_Setup(image_list, model_name="vgg19", pretrained=True)
print("Model Loaded Successfully: vgg19")

# Define Flask routes
@app.route("/", methods=["POST", "GET"])
def new():
    return jsonify({"Testing": "Hello"})

@app.route("/api", methods=["POST", "GET"])
def index():
    print("API endpoint hit")
    if request.method == "POST":
        print("Received POST request")
        image = request.files.get("fileup")
        if image:
            try:
                newimage = Image.open(image)
                newimage.save("uploaded_image.jpg")
                similar_images = st.get_similar_images(
                    image_path="uploaded_image.jpg", number_of_images=10
                )
                images = [similar_images[index] for index in similar_images]
                return jsonify({"similar_images": images})
            except Exception as es:
                return jsonify({"error": str(es)}), 400
        else:
            return jsonify({"Error": "Invalid image file."}), 400
    else:
        return jsonify({"Error": "No image provided."}), 400

# Function to run Flask in a thread
def run_flask():
    app.run(debug=True, host="0.0.0.0", port=5000, use_reloader=False)

# Start the Flask app in a new thread
flask_thread = threading.Thread(target=run_flask)
flask_thread.start()


# Function to run Flask in a thread
# def run_flask():
#     app.run(debug=True, host="127.0.0.1", port=5000, use_reloader=False)

