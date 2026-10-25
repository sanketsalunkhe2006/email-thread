from pathlib import Path
import json, sys
import joblib
import pandas as pd
from sklearn.pipeline import Pipeline
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, classification_report

ROOT=Path(__file__).resolve().parent
DATA=ROOT/"data"/"emails.csv"
OUT=ROOT/"models"
def main():
    if not DATA.exists():
        print(f"Dataset missing: {DATA}"); sys.exit(1)
    df=pd.read_csv(DATA)
    if not {"text","label"}.issubset(df.columns):
        raise ValueError("CSV must contain columns named text and label.")
    df=df.dropna(subset=["text","label"]).copy()
    df["text"]=df["text"].astype(str).str.strip()
    df["label"]=df["label"].astype(str).str.lower().str.strip()
    df=df[df["label"].isin(["phishing","legitimate"])]
    df=df[df["text"]!=""].drop_duplicates(subset=["text","label"])
    if len(df)<10 or df["label"].nunique()<2:
        raise ValueError("Need at least 10 examples and both labels: phishing and legitimate.")
    Xtr,Xte,ytr,yte=train_test_split(df.text,df.label,test_size=.25,random_state=42,stratify=df.label)
    model=Pipeline([("tfidf",TfidfVectorizer(lowercase=True,strip_accents="unicode",ngram_range=(1,2),max_features=12000,sublinear_tf=True)),
                    ("classifier",LogisticRegression(max_iter=1500,class_weight="balanced",random_state=42))])
    model.fit(Xtr,ytr); pred=model.predict(Xte)
    m={"accuracy":float(accuracy_score(yte,pred)),
       "precision":float(precision_score(yte,pred,pos_label="phishing",zero_division=0)),
       "recall":float(recall_score(yte,pred,pos_label="phishing",zero_division=0)),
       "f1_score":float(f1_score(yte,pred,pos_label="phishing",zero_division=0)),
       "train_rows":int(len(Xtr)),"test_rows":int(len(Xte)),"dataset":"data/emails.csv"}
    OUT.mkdir(exist_ok=True)
    joblib.dump(model,OUT/"email_model.joblib")
    (OUT/"metrics.json").write_text(json.dumps(m,indent=2),encoding="utf-8")
    print("Training complete. Model and metrics saved in models/.")
    print(json.dumps(m,indent=2)); print(classification_report(yte,pred,zero_division=0))
if __name__=="__main__": main()
