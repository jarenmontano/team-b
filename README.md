# GradingAPI

How to get it up and running

1. use the download in vs code with the link
this will download the project then in your terminal while in the grading api folder.


# pip install -r requirements.txt

in your terminal


then flask run within the soccerapiv1 you need to activate the python virtual environment or set up your own.

once it is activated with in that terminal use this command
flask run



if you get this error 
Error: Could not locate a Flask application. Use the 'flask --app' option, 'FLASK_APP' environment variable, or a 'wsgi.py' or 'app.py' file in the current directory

you need run this line

this is telling it where to start running so if you chose to not use a virtual python environment just make sure that the file path is correct for your use based on where your terminal is located with in your file structure(File Explorer or Finder).

export FLASK_APP=../app.py
