from flask import Flask, request, jsonify, render_template, session
import numpy as np
import pandas as pd
import joblib
import os
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.linear_model import LinearRegression




Xscaler = joblib.load('model\Xscaler_Advert.pkl')
Yscaler = joblib.load('model\Yscaler_Advert.pkl')
model=joblib.load('model\Advert_model.pkl')


app = Flask(__name__)
app.secret_key =os.urandom(24)

@app.route('/') 
def ad_home():
    return render_template('ad_home.html')



@app.route('/predict', methods=['POST'])
def predict():
    TV=request.form['TV']
    Billboards=request.form['Billboards']
    Google_Ads=request.form['Google_Ads']
    Social_Media=request.form['Social_Media']
    Influencer_Marketing=request.form['Influencer_Marketing']
    Affiliate_Marketing=request.form['Affiliate_Marketing']

    info={'TV':TV,
       'Billboards':Billboards,	
       'Google_Ads':Google_Ads,
       'Social_Media':Social_Media,	
       'Influencer_Marketing':Influencer_Marketing,	
       'Affiliate_Marketing':Affiliate_Marketing
      }
    
    info=pd.DataFrame(info, index=[0])
    predictor=info.to_numpy()
    predictor=Xscaler.transform(predictor)

    prediction=model.predict(predictor)
    prediction=Yscaler.inverse_transform(prediction)

    return render_template('ad_result.html', 
                           TV=TV, Billboards=Billboards, 
                           Google_Ads=Google_Ads, 
                           Social_Media=Social_Media, 
                           Influencer_Marketing=Influencer_Marketing,
                           Affiliate_Marketing=Affiliate_Marketing, 
                           prediction=prediction[0][0])

if __name__ == "__main__":
    app.run(host='0.0.0.0', port=81)

