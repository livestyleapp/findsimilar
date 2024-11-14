from flask import Flask, jsonify, request
import os
from PIL import Image
from DeepImageSearch import Load_Data, Search_Setup
import threading

# Ensure model weights are available locally
weights_folder = "metadata-files/vgg19"
weights_files = ["image_features_vectors.idx", "image_data_features1.pkl"]

# Create the folder if it doesn't exist
os.makedirs(weights_folder, exist_ok=True)

# Check if the weights files exist locally
if not all(os.path.exists(os.path.join(weights_folder, wf)) for wf in weights_files):
    print("Weights files are missing. Please ensure they are available locally.")
else:
    print("Using local weights files.")

# Load data only once
print("Loading Data")
image_list = Load_Data().from_folder(folder_list=["Data"])
print("Data Loaded")
print(f"Number of images loaded: {len(image_list)}")

app = Flask(__name__)

# Global variable for Search_Setup
st = None

def is_valid_image(file):
    try:
        image = Image.open(file)
        image.verify()
        return True
    except Exception as e:
        print(f"Invalid image: {e}")
        return False

def load_model():
    global st
    print("Loading Model in Background")
    st = Search_Setup(image_list, model_name="vgg19", pretrained=True)
    st.get_image_metadata_file()  # Load metadata once here
    print("Model and Metadata Loaded Successfully: vgg19")

# Start model loading in a separate thread
threading.Thread(target=load_model).start()

@app.route('/', methods=['POST', "GET"])
def new():
    print("Received request on /")
    return jsonify({"Testing": "Hello"})

@app.route('/api', methods=['POST', "GET"])
def index(): 
    if request.method == 'POST':
        print("Received POST request on /api")
        image = request.files.get('fileup')
        if not image:
            print("No file uploaded")
            return jsonify({"Error": "No file uploaded"}), 400

        if not is_valid_image(image):
            print("Uploaded file is not a valid image")
            return jsonify({"Error": "Invalid image file"}), 400

        newimage = Image.open(image)
        print("Image Loaded")
        newimage.save("Ahmed.jpg")
        print("Image Saved as Ahmed.jpg")

        try:
            # Retrieve similar images without reloading metadata
            similar_images = st.get_similar_images(image_path="Ahmed.jpg", number_of_images=10)
            print("Similar images found")
            images = [similar_images[index] for index in similar_images]
            print("Images processed")
            return jsonify({"Testing": images})
        except Exception as e:
            print(f"Error processing image: {e}")
            return jsonify({"Error": "Failed to process image"}), 500
    else:
        print("Received GET request on /api")
        return jsonify({"Error": "No Images"})

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    app.run(debug=True, host="0.0.0.0", port=port, use_reloader=False)
