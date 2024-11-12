from flask import Flask, jsonify, request, g
import os
from PIL import Image
from DeepImageSearch import Load_Data, Search_Setup
import wget
import shutil
import threading

# Ensure model weights are downloaded only if they don't already exist
weights_folder = "metadata-files/vgg19"
weights_files = ["image_features_vectors.idx", "image_data_features.pkl"]

# Create the folder if it doesn't exist
os.makedirs(weights_folder, exist_ok=True)

# Download the weights if they're missing
if not all(os.path.exists(os.path.join(weights_folder, wf)) for wf in weights_files):
    print("Downloading Weights")
    wget.download("https://download1326.mediafire.com/1ga0rvjg6qkgha8oOPFVHY2V_WLYwde6gCWBXT1FQlhATeqS-GNBaZrzcuNuwZSCLs76qScw1vWYEhFSLNiKLjkMJX3Gog3gCSUz2VzV2IRE5ZfZOySSbF5HHww8AU9_dYJNZIriOyGNI4hTosvysfJZvpOY93jRC2z77H9N_66D/ekg10hcq141v5hi/image_features_vectors.idx")
    wget.download("https://download1584.mediafire.com/29g7k41bwpqg5eCW0sOQnrNX853Xxi74F66GoMquaXqTH4jeAPyMKr4H8N2vrtBfjscyLDHOJxyMusdAfZEMRq7ZbKzG3hKPPbjf_YIDPcgAf3Keca8UzD6kAPro24YbwhV_63uLYfYWAG9ka8c-L_rCFZ09alLyplZ3OTS_uKfC/0aemngb1qh3t9cn/image_data_features.pkl")
    print("Weights Downloaded")
    shutil.move("image_features_vectors.idx", os.path.join(weights_folder, "image_features_vectors.idx"))
    shutil.move("image_data_features.pkl", os.path.join(weights_folder, "image_data_features.pkl"))
else:
    print("Weights already exist. Skipping download.")

# Load data
print("Loading Data")
image_list = Load_Data().from_folder(folder_list=["Data"])
print("Data Loaded")
print(f"Number of images loaded: {len(image_list)}")

app = Flask(__name__)

def is_valid_image(file):
    try:
        image = Image.open(file)
        image.verify()
        return True
    except:
        return False

@app.route('/', methods=['POST', "GET"])
def new():
    return jsonify({"Testing": "Hello"})

@app.route('/api', methods=['POST', "GET"])
def index():
    if 'st' not in g:
        print("Loading Model")
        g.st = Search_Setup(image_list, model_name="vgg19", pretrained=True)
        print("Model Loaded Successfully: vgg19")
    
    if request.method == 'POST':
        print("Post here")
        image = request.files['fileup']
        newimage = Image.open(image)
        print("Image Loaded")
        newimage.save("Ahmed.jpg")
        print("Image Saved")
        print("we are here")
        x = g.st.get_image_metadata_file()
        print("metadata loaded")
        similar_images = g.st.get_similar_images(image_path="Ahmed.jpg", number_of_images=10)
        print("similar images")
        images = [similar_images[index] for index in similar_images]
        print("Images are done")
        return jsonify({"Testing": images})
    else:
        print("we are in else")
        return jsonify({"Error": "No Images"})

def run_flask():
    port = int(os.environ.get("PORT", 5000))
    app.run(debug=True, host="0.0.0.0", port=port, use_reloader=False)

# Start the Flask app in a new thread
flask_thread = threading.Thread(target=run_flask)
flask_thread.start()