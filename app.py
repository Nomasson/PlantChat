from flask import Flask, request, jsonify, render_template,session, redirect, url_for, make_response
from werkzeug.utils import secure_filename
import json, pickle, requests, os, base64, nltk, random
import tensorflow as tf
from nltk.stem import WordNetLemmatizer
lemmatizer = WordNetLemmatizer()
import numpy as np
from tensorflow.keras.models import load_model
model = load_model('chatbot_model.h5')

# import requests
# import os, base64
# import nltk
# import pickle
# import random

intents = json.loads(open('intents.json').read())
words = pickle.load(open('words.pkl','rb'))
classes = pickle.load(open('classes.pkl','rb'))

def allowed_image_filesize(filesize):

    if int(filesize) <= app.config["MAX_IMAGE_FILESIZE"]:
        return True
    else:
        return False

def allowed_image(filename):
    """
    validate image extension
    """
    if not "." in filename:
        return False

    ext = filename.rsplit(".", 1)[1]

    if ext.upper() in app.config["ALLOWED_IMAGE_EXTENSIONS"]:
        return True
    else:
        return False

def send_image(image):
    json_data = {
        "images": image,
        "modifiers": ["similar_images"],
        "plant_details": ["common_names", "url", "wiki_description", "taxonomy"]
    }

    response = requests.post(
        "https://api.plant.id/v2/identify",
        json=json_data,
        headers={
            "Content-Type": "application/json",
            "Api-Key": app.config["API_KEY"]
        }).json()
    if response["suggestions"] is not None:
        return response
    else:
        return False
def clean_up_sentence(sentence):
    sentence_words = nltk.word_tokenize(sentence)
    sentence_words = [lemmatizer.lemmatize(word.lower()) for word in sentence_words]
    return sentence_words

# return bag of words array: 0 or 1 for each word in the bag that exists in the sentence

def bow(sentence, words, show_details=True):
    # tokenize the pattern
    sentence_words = clean_up_sentence(sentence)
    # bag of words - matrix of N words, vocabulary matrix
    bag = [0]*len(words)
    for s in sentence_words:
        for i,w in enumerate(words):
            if w == s:
                # assign 1 if current word is in the vocabulary position
                bag[i] = 1
                if show_details:
                    print ("found in bag: %s" % w)
    return(np.array(bag))

def predict_class(sentence, model):
    # filter out predictions below a threshold
    p = bow(sentence, words,show_details=False)
    res = model.predict(np.array([p]))[0]
    ERROR_THRESHOLD = 0.25
    results = [[i,r] for i,r in enumerate(res) if r>ERROR_THRESHOLD]
    # sort by strength of probability
    results.sort(key=lambda x: x[1], reverse=True)
    return_list = []
    for r in results:
        return_list.append({"intent": classes[r[0]], "probability": str(r[1])})
    return return_list

def getResponse(ints, intents_json):
    tag = ints[0]['intent']
    list_of_intents = intents_json['intents']
    for i in list_of_intents:
        if(i['tag']== tag):
            result = random.choice(i['responses'])
            break
    return result
def send(massage='massage'):

    msg = massage
    res = chatbot_response(msg)
    return res


def chatbot_response(msg):
    ints = predict_class(msg, model)
    res = getResponse(ints, intents)
    return res

app = Flask(__name__,template_folder="templates", instance_relative_config=True)


app.config.from_object('config')
app.config.from_pyfile('config.py')

@app.route('/', methods=['GET', 'POST'])
def upload_image():
    error = ""
    session['session_id']='12345678'
    if request.method == "POST":


        if request.files:

            if "filesize" in request.cookies:

                if not allowed_image_filesize(request.cookies["filesize"]):
                    error= "Filesize exceeded maximum limit"
                    print("Filesize exceeded maximum limit")
                    return redirect(request.url)

                image = request.files["image"]

                if image.filename == "":
                    error= "No fileName"
                    print("No filename")
                    return redirect(request.url)

                if allowed_image(image.filename):

                    filename = secure_filename(image.filename)
                    image.save(os.path.join(app.config["IMAGE_UPLOADS"], filename))

                    with open(app.config["IMAGE_UPLOADS"]+"/"+filename, "rb") as file:
                        image = [base64.b64encode(file.read()).decode("ascii")]

                        
                        # response = send_image(image)
                        
                        session['plant_name'] = 'roses'
                        # session['info'] = response['plant_details']['wiki_description']['value']
                        # for suggestion in response["suggestions"]:
                        #     print(suggestion["plant_name"])
                        #     print(suggestion["plant_details"]["common_names"])
                        #     print(suggestion["plant_details"]["url"])
                        return render_template('home.html',plant_text="plant_text")
                        # return redirect(url_for('/bot'), code =307)

                else:
                    print("That file extension is not allowed")
                    return redirect(request.url)

    return render_template("upload_image.html", error=error)

@app.route('/send_message', methods=['POST'])
def send_message():
    # session['session_id']
    message = request.form['message']
    fulfillment_text = send(message)
    response_text = { "message":  fulfillment_text }

    return jsonify(response_text)
if __name__ == "__main__":
    app.run(threaded=True, debug=True)

