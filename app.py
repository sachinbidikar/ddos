from flask import Flask, render_template, request, flash, redirect, url_for
import warnings
import pandas as pd
import numpy as np
import pickle
import os
warnings.filterwarnings("ignore")

app = Flask(__name__)
app.secret_key = "secret key"


class_names = ['dos','normal','probe','r2l','u2r']

with open(file='model/AdaBoostClassifier.pkl',mode='rb') as file:
    model = pickle.load(file=file)

with open(file='model/Normalization_model.pkl', mode='rb') as file:
    scaler = pickle.load(file=file)


@app.route("/")
def index():
    return render_template("index.html")


@app.route('/back', methods=['POST','GET'])
def back():
    return render_template('index1.html')


@app.route("/login",methods=["GET", "POST"])
def login():
    if request.method == "POST":
        uname = request.form["uname"]
        passw = request.form["passw"]
        r1 = pd.read_excel('user.xlsx')
        for index, row in r1.iterrows(): 
            if row["u_name"] == str(uname) and row["pass"] == str(passw):
                return render_template("index1.html")
        msg = "Invalid credentials"
        return render_template("login.html", msg=msg)
    return render_template("login.html")


@app.route("/register", methods=["GET", "POST"])
def register():
    if request.method == "POST":
        uname = request.form['uname']
        passw = request.form['passw']

        r1 = pd.read_excel('user.xlsx')
        new_row = {'u_name':uname,'pass':passw}
        r1 = r1.append(new_row, ignore_index=True)

        r1.to_excel('user.xlsx')
        
        print("Records created successfully")
        msg = "Hii " + uname + " Login here...."
        return render_template("login.html", msg=msg)
    return render_template("register.html")


def res():
    df = pd.read_excel('upload/test_data.xlsx')
    org_df = df.copy()
    df.head()
    scaled_data = scaler.transform(df.values)
    model_pred = model.predict(scaled_data)
    print(model_pred)
    df['Result'] = model_pred
    df['Result'] = df['Result'].apply(lambda x: 'BENIGN' if x == 0 else 'DDOS')
    df.to_csv("Result.csv", index=False)
    input_cols = org_df.columns
    output_cols = df.columns
    input_values = np.asarray(org_df)
    output_values = np.asarray(df)

    # values = df.values
    return input_cols, output_cols, input_values, output_values


@app.route('/predict', methods=['POST','GET'])
def predict():
    if request.method == 'POST':
        file1 = request.files['excel']
        file1.save('upload/test_data.xlsx')
        result = res()
        return render_template("result1.html", prediction=True, i_cols=result[0], o_cols=result[1], i_values=result[2],
                               o_values=result[3])


@app.route('/logout', methods=['POST','GET'])
def user_logout():
    return render_template('index.html')


if __name__ == '__main__':
    app.run(debug=True)
