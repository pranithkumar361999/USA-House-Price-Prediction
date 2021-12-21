from flask import Flask,render_template,request,url_for
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.model_selection import ShuffleSplit
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import GridSearchCV
from sklearn.linear_model import LinearRegression
from sklearn.linear_model import Lasso
from sklearn.tree import DecisionTreeRegressor
from werkzeug.utils import redirect
app = Flask(__name__)

@app.route("/")
def index():
    return render_template("home.html")

@app.route("/predict",methods=['GET','POST'])
def form():
    
    if request.method=="POST":
        data=pd.read_csv("https://raw.githubusercontent.com/pranithkumar361999/USA-House-Price-Prediction/main/USA_Housing.csv")
        data=data.drop(['Address'],axis='columns')
        X=data.drop("Price",axis=1)
        Y=data["Price"]
        X_train,X_test,Y_train,Y_test=train_test_split(X,Y,test_size=.30)
        model=LinearRegression()
        model.fit(X_train,Y_train)
        var1=float(request.form['avg-income'])
        var2=float(request.form['avg-age'])
        var3=float(request.form['avg-rooms'])
        var4=float(request.form['avg-bedrooms'])
        var5=float(request.form['avg-population'])
        pred=model.predict(np.array([var1,var2,var3,var4,var5]).reshape(1,-1))
        pred=round(pred[0])
        price="The Predicted Price is $"+str(pred)
        s=model.score(X_test,Y_test)*100
        accuracy="The Accuracy is "+str(s)+"%"
        return render_template("predict.html",price=price,accuracy=accuracy)
    else:
        return render_template("predict.html")



    

if __name__=="__main__":
    app.run(debug=True)