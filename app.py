from flask import Flask, request, render_template
from keras.models import load_model
from PIL import Image, ImageOps
import numpy as np
import base64
import re
import io

app = Flask(__name__)

model = load_model("model/model.h5")

def preprocess_digit(image):
    inverted = ImageOps.invert(image)
    img_array = np.array(inverted).astype("float32") / 255.0
    threshold = 0.1
    binary = img_array > threshold
    coords = np.column_stack(np.where(binary))
    if coords.size > 0:
        y0, x0 = coords.min(axis=0)
        y1, x1 = coords.max(axis=0)
        cropped = inverted.crop((x0, y0, x1 + 1, y1 + 1))
        resized = cropped.resize((20, 20), Image.LANCZOS)
    else:
        resized = inverted.resize((20, 20), Image.LANCZOS)
    new_image = Image.new("L", (28, 28))
    offset = ((28 - 20) // 2, (28 - 20) // 2)
    new_image.paste(resized, offset)
    final_array = np.array(new_image).astype("float32") / 255.0
    final_array = final_array.reshape(1, 28, 28, 1)
    return final_array

@app.route("/", methods=["GET", "POST"])
def index():
    prediction = None
    if request.method == "POST":
        data_url = request.form.get("image")
        if data_url:
            img_str = re.search(r'base64,(.*)', data_url).group(1)
            img_bytes = base64.b64decode(img_str)
            image = Image.open(io.BytesIO(img_bytes)).convert("L")
            processed_image = preprocess_digit(image)
            prediction = model.predict(processed_image).argmax()
    return render_template("index.html", prediction=prediction)

if __name__ == "__main__":
    app.run(debug=True)