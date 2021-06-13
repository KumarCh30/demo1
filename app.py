import numpy as np
import pandas as pd
from flask import Flask, request, render_template
import pickle

app = Flask(__name__)
model = pickle.load(open('model.pkl', 'rb'))

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict',methods=['POST'])
def predict():
    input_features = [float(x) for x in request.form.values()]
    features_value = [np.array(input_features)]
    
    features_name = [ "having_IP_Address", "URL_Length", "Shortining_Service", "having_At_Symbol",
                     "double_slash_redirecting", "Prefix_Suffix", "having_Sub_Domain", "SSLfinal_State",
                    "Domain_registeration_length", "Favicon", "port", "HTTPS_token", "Request_URL",
                    "URL_of_Anchor", "Links_in_tags", "SFH", "Submitting_to_email", "Abnormal_URL",
                    "Redirect", "on_mouseover", "RightClick", "popUpWidnow", "Iframe", "age_of_domain",
                     "DNSRecord", "web_traffic", "Page_Rank", "Google_Index", "Links_pointing_to_page",
                     "Statistical_report"]
    
    df = pd.DataFrame(features_value, columns=features_name)
    output = model.predict(df)
        
    if output == 1:
        res_val = "** Phising Attack **"
    else:
        res_val = "no Attack "
        

    return render_template('index.html', prediction_text='Request is {}'.format(res_val))

if __name__ == "__main__":
    app.run()
