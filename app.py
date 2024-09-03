import datetime
import textwrap

from flask import Flask, render_template, request, jsonify
import requests
import json

from decisiontreeregmodel import predict_student_score, submit_feedback

# global variables
results = ""
sk_results = ""

app = Flask(__name__)

@app.route("/", methods = ['GET'])
def index():
    global results
    return "Server is running"

@app.route("/predictioncenter", methods = ['POST'])
def get_model_response():
    print(request.json)
    content = request.json
    student_resposne = content['student_response']
    correct_answer=content['correct_answer']
    rubric=content['rubric']
    max_possible_score=content['max_possible_score']
    
    
    response = predict_student_score(student_response=student_resposne, correct_answer=correct_answer, max_possible_score=max_possible_score, rubric=rubric)      
    
    return jsonify(response)


@app.route("/modelfeedback", methods = ['POST'])
def model_feedback():
    print(request.json)
    content = request.json
    student_response = content['student_response']
    correct_answer=content['correct_answer']
    rubric=content['rubric']
    max_possible_score=content['max_possible_score']
    user_chosen_score=content['user_chosen_score']
    temp_response = { "content_recieved": content,
                     "response" : "Model has Failed Training"
    }
    
    submit_feedback_response = submit_feedback(student_response=student_response, correct_answer=correct_answer, rubric=rubric, max_possible_score=max_possible_score, user_chose_score=user_chosen_score)
    if(submit_feedback_response):
        temp_response = { "content_recieved": content,
                     "response" : "Model has Trained succesfully"
        }
        return temp_response

    return temp_response
    


if __name__ == "__main__":
    app.run(debug=True)



    