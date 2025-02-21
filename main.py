import os
import time
import pickle
import numpy as np
import pefile
import lightgbm as lgb
from fastapi import FastAPI, File, UploadFile
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from lightgbm import LGBMClassifier, early_stopping, log_evaluation

def extract_pe_features(file_path):
    try:
        pe = pefile.PE(file_path)
        features = {
            "num_sections": len(pe.sections),
            "num_imports": sum(len(entry.imports) for entry in pe.DIRECTORY_ENTRY_IMPORT) if hasattr(pe, 'DIRECTORY_ENTRY_IMPORT') else 0,
            "image_base": pe.OPTIONAL_HEADER.ImageBase,
            "size_of_code": pe.OPTIONAL_HEADER.SizeOfCode,
            "size_of_headers": pe.OPTIONAL_HEADER.SizeOfHeaders,
            "entry_point": pe.OPTIONAL_HEADER.AddressOfEntryPoint,
        }
        return np.array(list(features.values())).reshape(1, -1)
    except Exception as e:
        print(f"Error extracting features: {e}")
        return None

def load_real_ransomware_data():
    X, Y = [], []
    ransomware_dir = "./data/ransomware/"
    normal_dir = "./data/normal/"
    
    for file in os.listdir(ransomware_dir):
        features = extract_pe_features(os.path.join(ransomware_dir, file))
        if features is not None:
            X.append(features[0])
            Y.append(1)
    
    for file in os.listdir(normal_dir):
        features = extract_pe_features(os.path.join(normal_dir, file))
        if features is not None:
            X.append(features[0])
            Y.append(0)
    
    return np.array(X), np.array(Y)

X_real, y_real = load_real_ransomware_data()
X_train, X_test, y_train, y_test = train_test_split(X_real, y_real, test_size=0.2, random_state=42)

model = LGBMClassifier(
    objective='binary',
    num_leaves=31,
    learning_rate=0.05,
    n_estimators=100
)

model.fit(
    X_train, y_train,
    eval_set=[(X_test, y_test)],
    eval_metric='logloss',
    callbacks=[early_stopping(10), log_evaluation(1)]
)

with open("model.pkl", "wb") as f:
    pickle.dump(model, f)

app = FastAPI()

@app.post("/upload/")
async def upload_file(file: UploadFile = File(...)):
    start_time = time.time()
    file_path = f"temp_{file.filename}"
    
    with open(file_path, "wb") as f:
        f.write(await file.read())
    
    features = extract_pe_features(file_path)
    os.remove(file_path)
    
    if features is None:
        return {"error": "Invalid PE file"}
    
    with open("model.pkl", "rb") as f:
        model = pickle.load(f)
    
    prediction = model.predict(features)
    label = "Ransomware" if prediction[0] == 1 else "Normal"
    elapsed_time = time.time() - start_time
    
    return {"prediction": label, "execution_time": elapsed_time}
