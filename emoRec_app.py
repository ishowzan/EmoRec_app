import os
from flask import Flask, request, redirect, render_template, flash, send_from_directory
from werkzeug.utils import secure_filename
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
import numpy as np
from PIL import Image

classes = ["Angry", "Disgust", "Fear", "Happy", "Sad", "Surprise", "Neutral"]
image_size = 48
UPLOAD_FOLDER = "uploads"
ALLOWED_EXTENSIONS = set(['png', 'jpg', 'jpeg', 'gif'])

import tensorflow as tf
print("TensorFlow Version:", tf.__version__)

app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

model = load_model('./emotion_model.h5', compile=False)
model.compile(optimizer='adadelta', loss='categorical_crossentropy', metrics=['accuracy'])

@app.route('/', methods=['GET', 'POST'])
def upload_file():
    if request.method == 'POST':
        if 'file' not in request.files:
            flash('ファイルがありません')
            return redirect(request.url)
        file = request.files['file']
        if file.filename == '':
            flash('ファイルがありません')
            return redirect(request.url)
        if file and allowed_file(file.filename):
            filename = secure_filename(file.filename)
            file.save(os.path.join(UPLOAD_FOLDER, filename))
            filepath = os.path.join(UPLOAD_FOLDER, filename)

            # 48x48のグレイスケール画像に変換
            with Image.open(filepath) as img:
                img = img.convert("L") # モノクロに変換
                img = img.resize((image_size, image_size))
        
                # この加工された画像をprocessed_のプレフィックスをつけて保存
                processed_filepath = os.path.join(UPLOAD_FOLDER, "processed_" + filename)
                img.save(processed_filepath)

            img_array = image.img_to_array(img)


            # 画像の正規化
            img_array = img_array / 255.0
            
            data = np.array([img_array])

            result = model.predict(data)[0]
            print(f"result = {result}")
            
            predicted = result.argmax()
            pred_answer = "感情は " + classes[predicted] + " です"

            # 予測結果を小数点以下3桁で取得
            probabilities = ["{:.3f}".format(value) for value in result]
            
            # classesとprobabilitiesをzipで組み合わせる
            class_probabilities = list(zip(classes, probabilities))
            print( f"class_probabilities = {class_probabilities}" )

            return render_template("index.html", answer=pred_answer, input_image=filename, processed_image="processed_" + filename, class_probabilities=class_probabilities)

    return render_template("index.html", answer="", input_image=None, processed_image=None, class_probabilities=None)

@app.route('/uploads/<filename>')
def send_file(filename):
    return send_from_directory(UPLOAD_FOLDER, filename)

if __name__ == "__main__":
    port = int(os.environ.get('PORT', 8080))
    app.run(host='0.0.0.0', port=port)



"""
if __name__ == "__main__":
    app.run()
"""    


