import pickle as pic
import sys
#from sklearn.feature_extraction.text import TfidfVectorizer
#from sklearn.ensemble import RandomForestClassifier
vectorizer_runner=None
model_runner=None
def load_vectorizer():
    global vectorizer_runner
    vectorizer_runner=pic.load(open("data/song_vectorizer.pk", "rb"))
def load_model():
    global model_runner
    model_runner=pic.load(open("data/song_model.pk", "rb"))
def transformer(text):
    transformed=vectorizer_runner.transform(text)
    return transformed
def predicter(text_transformed, str1, str2):
    pred_str=""
    pred=model_runner.predict(text_transformed)
    if pred==0:
        pred_str=str1
    elif pred==1:
        pred_str=str2
    return pred_str
def calc_prob(text_transformed):
    prob=model_runner.predict_proba(text_transformed)
    if(prob[0][0]>prob[0][1]):
        return prob[0][0]
    return prob[0][1]
if __name__ == "__main__":
    if(len(sys.argv)>2):
        if(sys.argv[1]=="-t"):
            load_vectorizer()
            load_model()
            if(sys.argv[2]!=None):
                sample=[sys.argv[2]]
                sample_transformed=transformer(sample)
                result_str=predicter(sample_transformed,"Selena Gomez", "Justin Bieber")
                result_prob=calc_prob(sample_transformed)
                print("The model predicts that the sample text belongs to :"+ result_str+",\n with a probability of "+ str(result_prob*100)+"%")
            else:
                print("Error:  empty text")
    else:
        print("command usage error: <python_script_name>.py -t text")    
    
    
    
