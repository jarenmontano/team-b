import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split, RandomizedSearchCV
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, r2_score, confusion_matrix
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import re
import matplotlib.pyplot as plt
import seaborn as sns

# Load the dataset
file_path = r"/Users/jarenmontano/Documents/Internship/mlweb/gradingAPI/static/CleanedDataRavis.csv"
data = pd.read_csv(file_path)
#line to remove nan values from dataframe:
data.fillna(' ',inplace=True)
# Feature Engineering

#Creating another instance of data for the rag
rag_df= data[['StudentResponse', 'CorrectAnswer', 'MaxPossibleScore', 'StudentScore', 'Rubric' ]]


# 1. Cosine Similarity between Student Response and Correct Answer
tfidf_vectorizer = TfidfVectorizer()
tfidf_matrix = tfidf_vectorizer.fit_transform(data['StudentResponse'] + " " + data['CorrectAnswer'])

#vectorizing the student response for rag
vectorized_student_response = tfidf_vectorizer.transform(rag_df['StudentResponse'])
#For cosine_similarity
# print(vectorized_student_response[:5])

cosine_similarities = []
for i in range(len(data)):
    student_vector = tfidf_vectorizer.transform([data['StudentResponse'][i]])
    correct_vector = tfidf_vectorizer.transform([data['CorrectAnswer'][i]])
    cosine_similarity_score = cosine_similarity(student_vector, correct_vector)[0][0]
    cosine_similarities.append(cosine_similarity_score)
data['CosineSimilarity'] = cosine_similarities

# 2. Length of Student Response
data['ResponseLength'] = data['StudentResponse'].apply(len)

# 3. Word Count of Student Response
data['WordCount'] = data['StudentResponse'].apply(lambda x: len(x.split()))

# 4. Parse Rubric and Calculate Similarities
def parse_rubric(rubric_text):
    levels = {}
    matches = re.findall(r'(\d+):\s*(.*?)(?:\.|$)', rubric_text)
    for match in matches:
        level = int(match[0])
        description = match[1]
        levels[level] = description
    return levels

def rubric_similarity_features(row):
    rubric_levels = parse_rubric(row['Rubric'])
    student_response = row['StudentResponse']
    
    # Calculate TF-IDF similarity for each rubric level
    level_similarities = []
    for level, description in rubric_levels.items():
        combined_text = [student_response, description]
        similarity = cosine_similarity(tfidf_vectorizer.transform(combined_text))[0][1]
        level_similarities.append(similarity)
    
    # Return the highest similarity score
    return max(level_similarities) if level_similarities else 0

data['MaxRubricSimilarity'] = data.apply(rubric_similarity_features, axis=1)

# Splitting the data into training and testing sets
X = data[['CosineSimilarity', 'ResponseLength', 'WordCount', 'MaxRubricSimilarity']]
y = data['StudentScore']
print(X.head())
print(data.head())
print(rag_df.head())

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Model Training
model = RandomForestRegressor(random_state=42)
model.fit(X_train, y_train)

# Prediction
y_pred = model.predict(X_test)

# Evaluating the Model
mse = mean_squared_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)

print(f"Mean Squared Error: {mse}")
print(f"R^2 Score: {r2}")

# Adding Confusion Matrix
def print_confusion_matrix(y_true, y_pred):
    y_true_rounded = np.round(y_true)
    y_pred_rounded = np.round(y_pred)
    
    cm = confusion_matrix(y_true_rounded, y_pred_rounded)
    plt.figure(figsize=(10, 7))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
    plt.xlabel('Predicted')
    plt.ylabel('Actual')
    plt.title('Confusion Matrix')
    plt.show()

# print_confusion_matrix(y_test, y_pred)



def model_predict_student_score(student_response,correct_answer,rubric,max_possible_score):
    # Feature calculation for new input
    last_cosine_similarity = cosine_similarity(tfidf_vectorizer.transform([student_response]), tfidf_vectorizer.transform([correct_answer]))[0][0]
    last_response_length = len(student_response)
    last_word_count = len(student_response.split())
    last_max_rubric_similarity = rubric_similarity_features({'StudentResponse': student_response, 'Rubric': rubric})

    # Predict the score
    predicted_score = model.predict(np.array([[last_cosine_similarity, last_response_length, last_word_count, last_max_rubric_similarity]]))[0]
    print(f"Predicted Score: {predicted_score:.2f} / {max_possible_score}")

    return predicted_score





#****************************************************************
# functions 
def make_a_solid_prediction(student_response, correct_answer, max_possible_score, rubric):
    #This json should have maxpossiblescore, studentresponce, correctanswer, rubric
    predicted_score=model_predict_student_score(student_response,correct_answer,rubric,max_possible_score)

    return predicted_score

#Preprocessing before trying to get 
def predict_student_score(student_response, correct_answer,max_possible_score, rubric):
    print("Student Response: ",student_response)
    print("Correct Answer : " ,correct_answer)

    #checking if they are equal
    if(student_response == correct_answer):
        print("student_ response and correct_answer are equal. " + "-"*40)
        temp_response = {'predictedscore' : max_possible_score, "response" : "Student Response is equal to Correct Answer"}
        return temp_response
    #checking if student response is blank
    if(len(student_response) < 1):
        print("students response lenght was 0: " , student_response)
        temp_response = {'predictedscore' : max_possible_score, "response" : "Student Response is empty returning score as 0"}
        return temp_response
    # Trying to add embeddings to the StudentResponse column
    # cosine_similarity_score_output=cosine_similarity(
    #     tfidf_vectorizer.transform([student_response]),
    #     tfidf_vectorizer.transform([correct_answer])
    # )[0][0]


    # print(" Student Answer & Correct Answer\\n cosine Similarity score:", cosine_similarity_score_output)

    #Retrieving the index and similiarity score
    most_similiar_student_response_index , similiarity_score = retrieve_relevant_student_response(student_response, student_response_vectors=vectorized_student_response)
    
    # if the similarity is greater than 0.75 return full amount
    if( similiarity_score > 0.75):
        print("Cosign similarity score is greater than 75 %")
        print(rag_df.iloc[most_similiar_student_response_index])
        student_score = rag_df._get_value(most_similiar_student_response_index, 'StudentScore')
        print('Student Score from rag: ', student_score)
        temp_response = {'predictedscore' : student_score, "response" : "Using RAG to provide a response based on Previous answers"}
        return temp_response
    
    #Now doing the model prediction
    model_prediction = make_a_solid_prediction(student_response=student_response, correct_answer=correct_answer, rubric=rubric, max_possible_score=max_possible_score)
    if(model_prediction > max_possible_score):
            temp_response= {'predictedscore' : max_possible_score, "response" : "Using Decision Tree Model" }
            return temp_response
    temp_response= {'predictedscore' : model_prediction, "response" : "Using Decision Tree Model"}
    
    # LLM output is currently being commented out until

    # Call the LLM function instead of the standard model prediction
    # llm_output = get_llm_prediction(student_response=student_response, correct_answer=correct_answer, rubric=rubric)
            
    # # Prepare the response to return
    # temp_response = {
    #     'predictedscore': llm_output,  # The output from the LLM
    #     'response': 'Using LLM for prediction'
    # }
    return temp_response




    

#This will return the index of most similiar item
# will only return one index or the highest similarity
def retrieve_relevant_student_response(student_response, student_response_vectors):
    #Vectorizaing the query
    student_response_vector = tfidf_vectorizer.transform([student_response])
    #Compute Cosine Similarities
    similarities = cosine_similarity(student_response_vector, student_response_vectors).flatten()
    #Get the index of the most similar document
    most_similiar_student_response_index = np.argmax(similarities)
    similiarity_score = similarities[most_similiar_student_response_index]
    print(most_similiar_student_response_index)
    print("Similarity score : " , similiarity_score)
    return most_similiar_student_response_index , similiarity_score




#Retrains the model and historic dataset and the student response vectorized list.

def submit_feedback(student_response, correct_answer, rubric, max_possible_score, user_chose_score):
    try:
        
        feedback_cosine_similarity = cosine_similarity(tfidf_vectorizer.transform([student_response]), tfidf_vectorizer.transform([correct_answer]))[0][0]
        feedback_response_length = len(student_response)
        feedback_word_count = len(student_response.split())
        feedback_max_rubric_similarity = rubric_similarity_features({'StudentResponse': student_response, 'Rubric': rubric})

        new_data = {
                    'CosineSimilarity': feedback_cosine_similarity,
                    'ResponseLength': feedback_response_length,
                    'WordCount': feedback_word_count,
                    'MaxRubricSimilarity': feedback_max_rubric_similarity,
                    'StudentScore': user_chose_score
                }
        
        # Append the new data to the existing dataset
        global data, model
        new_df = pd.DataFrame([new_data])
        data = pd.concat([data, new_df], ignore_index=True)
                
        # Retrain the model with the updated dataset
        X = data[['CosineSimilarity', 'ResponseLength', 'WordCount', 'MaxRubricSimilarity']]
        y = data['StudentScore']
        model.fit(X, y)

        #Will need max possible score for adding to the historic dataset
        #When doing the Rag
        #This is where we will need to add to the historic dataset!
        global rag_df, vectorized_student_response
        rag_new_data ={
            'StudentResponse': student_response,
            'CorrectAnswer': correct_answer,
            'MaxPossibleScore': max_possible_score,
            'StudentScore': user_chose_score,
            'Rubric': rubric
        }

        

        rag_new_df = pd.DataFrame([rag_new_data])
        rag_df = pd.concat([ rag_df, rag_new_df ], ignore_index = True)
        # #Creating the new vectorized responses
        vectorized_student_response = tfidf_vectorizer.transform(rag_df['StudentResponse'])
        

        return True
    except:
        return False
    


# Function to interact with OpenAI LLM
# def get_llm_prediction(student_response, correct_answer, rubric):
#     import os
#     from openai import AzureOpenAI # type: ignore
    
    
#     # Assuming you have set up your OpenAI API key
#     client = AzureOpenAI(
#     api_key=os.getenv("AZURE_OPENAI_API_KEY"),  
#     api_version="2024-02-01",
#     azure_endpoint = os.getenv("AZURE_OPENAI_ENDPOINT")
#     )

#     deployment_name='REPLACE_WITH_YOUR_DEPLOYMENT_NAME' #This will correspond to the custom name you chose for your deployment when you deployed a model. Use a gpt-35-turbo-instruct deployment.
    
#     # Create the prompt for the LLM
#     prompt = f"""
#     Student Response: {student_response}
#     Correct Answer: {correct_answer}
#     Rubric: {rubric}
#     Based on the above, provide feedback or a revised score.
#     """
    
#     # Call the OpenAI LLM
#     response = client.completions.create(
#         engine="text-davinci-003",
#         prompt=prompt,
#         max_tokens=150
#     )
    
#     # Extract the LLM's response
#     llm_output = response.choices[0].text.strip()
    
#     return llm_output