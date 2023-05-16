model1=None
model2=None

def load_models():
    from tensorflow.keras.models import load_model

    global model1,model2
    model1 = load_model('Models/botnetANN.h5')
    model2= load_model('Models/botnetCNNLSTM.h5')

def process_data(path):
    import pandas as pd
    from pickle import load
    import numpy as np
    
    df = pd.read_csv(path, header=None)
    data = df.iloc[:1]
    scaler= load(open('Scaler/scaler.pkl', 'rb'))
    data = scaler.transform(data)

    return data


def get_prediction(path):
        
    import numpy as np

    input = process_data(path)

    map={ 
         0: 'Benign', 1: 'Mirai UDP', 2: 'Gafgyt COMBO', 3: 'Gafgyt Junk', 4: 'Gafgyt Scan', 5: 'Gafgyt TCP',
         6: 'Mirai ACK', 7: 'Mirai Scan', 8: 'Mirai SYN', 9: 'Mirai Plain UDP' }
    
    #from tensorflow.keras.models import load_model
    #model1 = load_model('Models/botnetANN.h5')
    #model2= load_model('Models/botnetCNNLSTM.h5')
    
    pred1 = model1.predict(input)
    pred1 = np.argmax(pred1,axis=1)
    prob1 = model1.predict_proba(input)
    prob1 = "%.2f" % (prob1[0][pred1]*100)
    
    input2 = np.reshape(input, (input.shape[0], input.shape[1],1))
    pred2 = model2.predict(input2)
    pred2 = np.argmax(pred2,axis=1)
    prob2 = model2.predict_proba(input2)
    prob2 = "%.2f" % (prob2[0][pred2]*100)

    if(pred1 != pred2):
        if(prob1 >= prob2):
            final_pred = pred1
        else:
            final_pred = pred2
    else:
        final_pred = pred1
    
    pred_label = map[final_pred[0]]
    return pred_label
    


