from flask import Flask, render_template, request, redirect, url_for, session
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import StandardScaler
from PIL import Image
import numpy as np
import re
import os
from sklearn.ensemble import  ExtraTreesRegressor
from flask import Flask, flash, request, redirect, url_for, render_template
from werkzeug.utils import secure_filename
import matplotlib.pyplot as plt
import numpy as np
import numpy as np
import os
from sklearn.model_selection import train_test_split
from flask import Flask, render_template, request, redirect, url_for, session
from flask_mysqldb import MySQL

from flask import Flask, render_template, request, redirect, url_for, session
from flask_mysqldb import MySQL
from flask import Flask,render_template,url_for,request
from flask_material import Material
import MySQLdb.cursors
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import GridSearchCV
from sklearn.naive_bayes import GaussianNB
from sklearn import svm
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
from sklearn import tree
from sklearn.svm import SVC
from sklearn.naive_bayes import GaussianNB
import matplotlib.pyplot as plt
import re
from flask_mysqldb import MySQL
from flask import Flask,render_template,url_for,request
import MySQLdb.cursors
import random
app = Flask(__name__)

# Enter your database connection details below
app.config['MYSQL_HOST'] = 'localhost'
app.config['MYSQL_USER'] = 'root'
app.config['MYSQL_PASSWORD'] = 'teju29@2003'
app.config['MYSQL_DB'] = 'healthy'
app.secret_key="dont tell any one"


mysql = MySQL(app)


@app.route('/')
def home():
    return render_template('index.html')



@app.route('/about')
def about():
    return render_template('about.html')


@app.route('/Predict')
def Predict():
    return render_template('Prediction.html')


from skimage import feature

@app.route('/login',methods=['GET', 'POST'])
def login():
    msg = ''

    if request.method == 'POST' and 'username' in request.form and 'password' in request.form:

        username = request.form['username']
        password = request.form['password']
        

        cursor = mysql.connection.cursor(MySQLdb.cursors.DictCursor)
        cursor.execute('SELECT * FROM user WHERE username = %s AND password = %s', (username, password))
        # Fetch one record and return result
        account = cursor.fetchone()
        print("accountttt",account)

        if account:

            session['loggedin'] = True
            session['id'] = account['id']
            session['username'] = account['username']
            # Redirect to home page
            return render_template('home.html')
        else:
            # Account doesnt exist or username/password incorrect
            msg = 'Incorrect username/password!'
    return render_template('Login.html', msg=msg)

# http://localhost:5000/pythinlogin/register - this will be the registration page, we need to use both GET and POST requests
@app.route('/register', methods=['GET', 'POST'])
def register():
    # Output message if something goes wrong...
    print("cllllllllllllllllllllllllll")
    msg = ''
    # Check if "username", "password" and "email" POST requests exist (user submitted form)
    if request.method == 'POST' and 'username' in request.form and 'password' in request.form and 'email' in request.form:
        # Create variables for easy access
        username = request.form['username']
        email = request.form['email']
        phone = request.form['phone']
                # Check if account exists using MySQL
        place = request.form['place']
        password = request.form['password']
        cursor = mysql.connection.cursor(MySQLdb.cursors.DictCursor)
        cursor.execute('SELECT * FROM user WHERE username = %s', (username,))
        account = cursor.fetchone()
        # If account exists show error and validation checks
        if account:
            msg = 'Account already exists!'
        elif not re.match(r'[^@]+@[^@]+\.[^@]+', email):
            msg = 'Invalid email address!'
        elif not re.match(r'[A-Za-z0-9]+', username):
            msg = 'Username must contain only characters and numbers!'
        elif not username or not password or not email:
            msg = 'Please fill out the form!'
        else:
            # Account doesnt exists and the form data is valid, now insert new account into accounts table
            cursor.execute('INSERT INTO user VALUES (NULL, %s, %s, %s,%s,%s)', (username, email, phone,place,password))
            mysql.connection.commit()
            msg = 'You have successfully registered!'
    elif request.method == 'POST':
        # Form is empty... (no POST data)
        msg = 'Please fill out the form!'
    # Show registration form with message (if any)
    return render_template('Register.html', msg=msg)


from flask import Flask, render_template, request
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier

@app.route('/predict', methods=["POST"])
def predict():
    if request.method == 'POST':
        gender = request.form['gender']
        age = request.form['age']
        weight = request.form['weight']
        blood_group = request.form['blood_group']
        bp = request.form['bp']
        hemo = request.form['hemo']
        sugar = request.form['sugar']
        platelets = request.form['platelets']
        wbc = request.form['wbc']
        rbc = request.form['rbc']
        hemocratic = request.form['hemocratic']
        cholesterol = request.form['cholesterol']
        LDL = request.form['LDL']
        HDL = request.form['HDL']
        triglycerides = request.form['triglycerides']
        protein = request.form['protein']

        age2 = 0
        weight2 = 0
        bp2 = 0
        sugar2 = 0
        hemo2 = 0
        platelets2 = 0
        red_blood_cell2 = 0

        # Clean the data by converting from unicode to float
        sample_data = [gender, age, weight, blood_group, bp, hemo, sugar, platelets, wbc, rbc]
        clean_data = [float(i) for i in sample_data]
        age1 = int(age)
        weight1 = int(weight)
        bp1 = int(bp)
        sugar1 = int(sugar)
        hemo1 = float(hemo)
        rbc1 = float(rbc)
        platelets1=int(platelets)

        if age1 < 18 or age1 > 50:
            age2 = age1
        if weight1 > 90:
            weight2 = weight1
        if bp1 > 120:
            bp2 = bp1
        if sugar1 > 120:
            sugar2 = sugar1
        if hemo1 < 13.5:
            hemo2 = hemo1
        if platelets1 < 150000:
            platelets2 = platelets1
        if rbc1 < 4.5:
            red_blood_cell2 = rbc1

        # Reshape the data as a sample, not individual features
        ex1 = np.array(clean_data).reshape(1, -1)

        # Load and process the dataset
        data = pd.read_csv('healthfood.csv')
        data = data.drop(columns=['name', 'id'])
        data = data.replace(
            ['female', 'male', 'Healthy', 'Not Healthy', 'A+', 'A-', 'B+', 'B-', 'AB+', 'AB-', 'O+', 'O-'],
            ['0', '1', '0', '1', '0', '1', '2', '3', '4', '5', '6', '7']
        )
        X = data.drop(columns=['eligibility'])
        y = data['eligibility'].values

        X_train1, X_test1, y_train1, y_test1 = train_test_split(X, y, test_size=0.2, random_state=1, stratify=y)


        clf = DecisionTreeClassifier(criterion="entropy", random_state=2, max_depth=5)
        clf.fit(X_train1, y_train1)



        result_clf = clf.predict(ex1)[0]


        class4 = "Not Healthy" if result_clf == '1' else "Healthy"



        Y_pred_dt = clf.predict(X_test1)
        score_dt = round(accuracy_score(Y_pred_dt, y_test1) * 100, 2)


        def predictedFood(res):
            if res == 1:
                disease = ["Diabetes", "Anxiety"]
                p_class = ["broccoli, Carrots, apple", "egg", "yogurt", "sunflower", "seeds"]
            elif res == 2:
                disease = ["Lupus", "Shingles"]
                p_class = ["EGGS", "Beans", "Greens", "Broccoli"]
            elif res == 3:
                disease = ["Herpes", "Pneumonia"]
                p_class = ["Beetroot", "Pomegranate", "Dates", "Banana"]
            elif res == 4:
                disease = ["Strep throat", "Bronchitis"]
                p_class = ["Eggs", "Kidney beans", "Brown", "rice Banana"]
            elif res == 5:
                disease = ["Anxiety", "Colds and Flu"]
                p_class = ["Citrus", "fruits", "Tomatoes", "Dark green leafy", "vegetables", "Berries"]
            elif res == 6:
                disease = ["Lyme disease", "Scabies"]
                p_class = ["Seafood", "Fruit", "juice", "Milk", "Brown Rice"]
            elif res == 7:
                disease = ["Retinal", "Colds and Flu"]
                p_class = ["Eggs", "Kidney beans", "Wheat", "Beetroot"]
            elif res == 8:
                disease = ["Conjunctivitis"]
                p_class = ["EGGS", "Beans", "Tomatoes", "Dark green leafy"]
            elif res == 9:
                disease = ["Retinal", "Pneumonia"]
                p_class = ["Tomatoes", "Dark green leafy", "vegetables", "Berries"]
            else:
                disease = ["Diarrhea", "Headaches"]
                p_class = ["Pomegranate", "EGGS", "Beans"]
            return p_class, disease

        df1 = pd.read_excel('bloodupdated.xlsx')
        X = df1.drop(columns=['class'])
        y = df1['class'].values
        X_train1, X_test1, y_train1, y_test1 = train_test_split(X, y, test_size=0.2, random_state=1, stratify=y)

        sample_data1 = [age, gender, weight, blood_group, bp, sugar, rbc, wbc, hemo, 332322, hemocratic, cholesterol, LDL, HDL, triglycerides, protein]
        clean_data1 = [float(i) for i in sample_data1]
        ex2 = np.array(clean_data1).reshape(1, -1)

        clf1 = DecisionTreeClassifier()
        clf1.fit(X_train1, y_train1)

        result_clf1 = clf1.predict(ex2)[0]
        class6, disease = predictedFood(result_clf1)

        youtube_urls = {
            "exercise_old": [
                "https://www.youtube.com/embed/ZKCSikhCH54",
                "https://www.youtube.com/embed/8BcPHWGQO44",
                "https://www.youtube.com/embed/HwES4OSc9H8",
                "https://www.youtube.com/embed/laIWV6qJWbk",
                "https://www.youtube.com/embed/IL3E0SGEWl0",
                "https://www.youtube.com/embed/Lco1LSrr2yM",
                
            ],
            "exercise_young": [
                "https://www.youtube.com/embed/qTHVnGA5rzU"
                "https://www.youtube.com/embed/-_VhU5rqyko",
                "https://www.youtube.com/embed/nJw9-15ncOM",
                "https://www.youtube.com/embed/Eaw_ObFm1EM",
                "https://www.youtube.com/embed/VVyEjBHiZOo",
                "https://www.youtube.com/embed/WpIFlh5whcs",
               
            ],
            "sugar": [
                "https://www.youtube.com/embed/OpSyWP7OL6s",
                "https://www.youtube.com/embed/aUqUbBzepWE",
                "https://www.youtube.com/embed/mfRGabWhBcQ",
                "https://www.youtube.com/embed/hRwzELaCHDA",
                "https://www.youtube.com/embed/TSnxM9DbppY",
                "https://www.youtube.com/embed/9OBA4Se99zs",
                
            ],
            "hemo": [
                "https://www.youtube.com/embed/5v91IgPtgSs",
                "https://www.youtube.com/embed/4rSJYxUWj5M",
                "https://www.youtube.com/embed/uEeco42iu7U",
                "https://www.youtube.com/embed/0i0PsMul7Dc",
                "https://www.youtube.com/embed/EFHCA0DAVCw",
     
            ],
            "platelets": [
                "https://www.youtube.com/embed/ee0LoPoa7gc",
                "https://www.youtube.com/embed/Xsk_8-Nm4vQ",
                "https://www.youtube.com/embed/CdI-pDLkxQM",
                "https://www.youtube.com/embed/d3KuEHCbIpY" ,
                "https://www.youtube.com/embed/z_gO3D630Go",
                "https://www.youtube.com/embed/Eqn4twKfYoI",
          
            ],
            "red_blood_cell": [
                "https://www.youtube.com/embed/qLWigi84qRs",
                "https://www.youtube.com/embed/DLwPMnFBwSA",
                "https://www.youtube.com/embed/dCJWaJrZSBA",
                "https://www.youtube.com/embed/hV5t4KKffKo",
                "https://www.youtube.com/embed/zFmBhaAUe4c",
                "https://www.youtube.com/embed/j9_PwLQ5peY",

            ],
            "age": [
                "https://www.youtube.com/embed/Lco1LSrr2yM",
                "https://www.youtube.com/embed/jbVT7GSDsEs"
                "https://www.youtube.com/embed/yQ0G5x5hI28" 
                "https://www.youtube.com/embed/MXiup0LHuTc",
               "https://www.youtube.com/embed/laIWV6qJWbk",
                "https://www.youtube.com/embed/IL3E0SGEWl0",
            ],
            "weight": [
                "https://www.youtube.com/embed/DRomQhvuFWA",
                "https://www.youtube.com/embed/pMKyVoMzQPE"
                "https://www.youtube.com/embed/digpucxFbMo",
                "https://www.youtube.com/embed/KqtWdZPbu1w",
                "https://www.youtube.com/embed/IsBR03WznCY",
                "https://www.youtube.com/embed/s6XgAhHNO2k",

            ]
        }

        # Select random YouTube URLs
        selected_urls = {key: random.choice(value) for key, value in youtube_urls.items()}

        return render_template('result.html', gender=gender, age=age, weight=weight, blood_group=blood_group, hemo=hemo, sugar=sugar,
                               wbc=wbc, rbc=rbc, result=result_clf1, age2=age2, weight2=weight2, bp2=bp2, sugar2=sugar2, hemo2=hemo2,
                               platelets2=platelets2, red_blood_cell2=red_blood_cell2,  class4=class4, score_dt=score_dt,  class6=class6,
                               disease=disease, selected_urls=selected_urls)

if __name__ == '__main__':
    app.run(debug=True)
