from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

# Imports
import fasttext

import flask
from flask import request
from flask import jsonify
import json 
import requests
import random

app = flask.Flask(__name__)

model_name = 'model'
train_file_name = 'train_data.txt'
response_file_name = 'story.json'

def train():
    print('--- training started ---')
    classifier = fasttext.supervised(train_file_name, model_name)
    print('--- training ended ---')
    return True

def predict(query):
    texts = [query]
    classifier = fasttext.load_model(model_name+'.bin')
    prediction = json.dumps(classifier.predict(texts, k=3))
    return prediction

def utter(prediction):
    response_file = open(response_file_name,'r')
    response = json.loads(response_file.read())
    prediction = json.loads(prediction)[0][0]
    utter_list = response[prediction]
    return random.choice(utter_list)

@app.route('/demo/predict/<query>',methods=['POST','GET'])
def ask(query):
    prediction = predict(query)
    utterance = utter(prediction)
    return utterance

@app.route('/demo/example',methods=['GET'])
def example():
    return '''
        [{
            "questions": ["hello", "Hi", "namaste"],
            "answers": ["hey", "hi"]
        }, 
        {
            "questions": ["how do you do", "How are you"],
            "answers": ["I' m good ", "Feeling great!"]
        }]
    '''

@app.route('/demo',methods=['POST','GET'])
def demo():
    if request.method == 'POST':
        QnAdata = json.loads(request.get_json()["data"])
        train_file = open(train_file_name,'w')
        response_file = open(response_file_name,'w')
        train_data = ''
        response_data = {}
        counter = 0
        for intent in QnAdata:
            questions = intent["questions"]
            answers = intent["answers"]
            for question in questions:
                train_data += '__label__QnA'+str(counter)+' '+question+'\n'
            response_data['__label__QnA'+str(counter)] = answers
            counter += 1
        train_file.write(train_data)
        response_file.write(json.dumps(response_data))
        train_file.close()
        response_file.close()
        train()
        return jsonify('{"status":"success"}')
    
    return '''
    <!doctype html>
    <html>
        <head>
            <meta charset="UTF-8">
            <meta name="viewport" content="width=device-width, initial-scale=1">
            <script src="https://code.jquery.com/jquery-3.3.1.min.js"></script>
            <script>
                function send(){
                    var textdata = $('#textArea').val()
                    $("#text").html('training agent..')
                    $.ajax({
                        type: 'POST',
                        url: '/demo',
                        data: JSON.stringify({"data":textdata}),
                        success: function(resp) {$("#text").html('training completed.')},
                        error: function (jqXHR, exception){console.log('error')},
                        contentType: "application/json",
                        dataType: 'json'
                    })
                }
                function test(){
                    var testarea = $('#testarea').val()
                    console.log(testarea)
                    $.ajax({
                        type: 'GET',
                        url: '/demo/predict/'+testarea,
                        success: function(resp) {$("#testresult").html(resp)},
                        error: function (jqXHR, exception){console.log('error')}
                    })
                }
            </script>
        </head>
        <body>
            <p> paste your FAQ here: <a href="/demo/example" target="new">see an example training data</a></p>
            <textarea id="textArea" rows="10"></textarea>
            <button onclick="send()">train</button>
            <p id="text"></p>
            <input type="text" id="testarea" placeholder="enter your query" />
            <button onclick="test()">test</button>
            <p id="testresult"></p>
        </body>
        <style>
            textarea,input {
                width: 99.7%;
                resize: none;
                border: 1px solid;
            }
            button {
                margin-top: 10px;
            }
        </style>
    </html>
    '''

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=80)
