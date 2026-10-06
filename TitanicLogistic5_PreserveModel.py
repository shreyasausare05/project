import pandas as pd
import numpy as np
import joblib 

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score,confusion_matrix

# Step 1: Load Data
#--------------------------------------------------------------
#   function name : Loaddata
#   Description : Load the data from CSV
#   Input  :     Name of Csv file
#   Output :     Data frame
#   Author name :Shreyas Ausare
#   Date :      16/08/2026
#--------------------------------------------------------------
def Loaddata(filename):
    df = pd.read_csv(filename)

    print("Dataset load succefully")
    print(df.head())

    return df

#Step 2: Preprocess Data
#--------------------------------------------------------------
#   function name : PreprocessData
#   Description : It perform data analysing
#   Input  :     Data frame
#   Output :     Updated Data frame
#   Author name :Shreyas Ausare
#   Date :      16/08/2026
#--------------------------------------------------------------


def PreprocessData(df):
    df = df.drop([
        "Passengerid",
        "zero",
        "name"
    ],
    errors = "ignore"
    )

    #Handle missing values
    df["Age"] = df["Age"].fillna(df["Age"].median())
    df["Fare"] = df["Fare"].fillna(df["Fare"].median())
    df["Embarked"] = df["Embarked"].fillna(df["Embarked"].median())

    #Convert catagorical to numeric data
    df = pd.get_dummies(
        df,
        columns=["Embarked"],
        drop_first=True,
        dtype=int
    )
    print(df.head())
    print("Data preprocessing completed")
    return df

#Step 3: Split data
#--------------------------------------------------------------
#   function name : Split data
#   Description : It perform spliting activity
#   Input  :     Data frame
#   Output :      Subset for trailing and testing
#   Author name :Shreyas Ausare
#   Date :      16/08/2026
#--------------------------------------------------------------
def Splitdata(df):
    X = df.drop("Survived",axis =1)
    Y = df["Survived"]

    X_train, X_test ,Y_train, Y_test = train_test_split(
        X,
        Y,
        test_size= 0.2,
        random_state= 42

    )
    return X_train,X_test,Y_train,Y_test

#Step 4: Train Model
#--------------------------------------------------------------
#   function name : Trained Model
#   Description : It perform Model training 
#   Input  :     Training ceatures  frame
#   Output :      Subset for trailing and testing
#   Author name :Shreyas Ausare
#   Date :      16/08/2026
#--------------------------------------------------------------

def TrainModel(X_train,Y_train):
    model = LogisticRegression (max_iter=1000)

    model = model.fit(X_train,Y_train)

    print("Model get trained")
    return model


# Step 5 : Evaluate model

#-----------------------------------------------------
#   Function Name : EvaluateModel
#   Description :   It performs model testing
#   Input :         model, testing data (fetures ,labels)
#   Output :        none
#   Author :        Shreyas Ausare
#   Date :          16/08/2026
#-----------------------------------------------------

def EvaulateModel(model,X_test,Y_test):
    Y_pred = model.predict(X_test)

    accuracy = accuracy_score(Y_test,Y_pred)

    print("Accuracy is :",accuracy *100)

    print(confusion_matrix(Y_test,Y_pred))

    return accuracy


# Step 6 : Preserve Model

#-----------------------------------------------------
#   Function Name : PreserveModel
#   Description :   It performs model preservation into .pkl file
#   Input :         model
#   Output :        none
#   Author :        Shreyas Ausare 
#   Date :          16/08/2026
#-----------------------------------------------------

def PreserveModel(model,filename):
    joblib.dump(model,filename)

    print("Model Preserve with name :",filename)

#--------------------------------------------------------------
#   function name : main
#   Description : Load the data from CSV
#   Input  :     Name of Csv file
#   Output :     Data frame
#   Author name :Shreyas Ausare
#   Date :      16/08/2026
#--------------------------------------------------------------

def main():
    #Step 1
    df =Loaddata("MarvellousTitanicDataset.csv")

    #Step 2
    df = PreprocessData(df)

    #Step 3
    X_train,X_test,Y_train,Y_test = Splitdata(df)

    # Step 4 :
    model = TrainModel(X_train,Y_train)

    #Step 5
    EvaulateModel(model,X_test,Y_test)

    #Step 6
    PreserveModel(model,"MarvellousTiatanic.pkl")

if __name__ == "__main__":
    main()