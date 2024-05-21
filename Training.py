"Step1 : Importing neccessary libraries"
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
pd.options.display.max_columns = 35
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import PowerTransformer
from sklearn.feature_selection import SelectFromModel
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
pip insatll xgboost
import xgboost as xgb
from sklearn.metrics import accuracy_score
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import balanced_accuracy_score
from sklearn.metrics import roc_curve, auc, precision_recall_curve, average_precision_score
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix, roc_auc_score, log_loss, mean_squared_error, mean_absolute_error
from sklearn.model_selection import GridSearchCV, StratifiedKFold


"Step 2 : Data slicing"

data = pd.read_csv("/content/gdrive/My Drive/MCA/MCA_rawdata.csv", parse_dates=["pxn_dt"])
test_data = data.copy()
test_data.drop(columns = ['APPLICATION.NO','CUSTOMER_NO', 'DPD', "Customer.Number"], inplace = True)


target_date1 = pd.to_datetime('2022-08-01')
target_date2 = pd.to_datetime('2022-09-01')
target_date3 = pd.to_datetime('2022-10-01')
target_date4 = pd.to_datetime('2022-11-01')


mask1 = test_data.pxn_dt == target_date1
mask2 = test_data.pxn_dt == target_date2
mask3 = test_data.pxn_dt == target_date3
mask4 = test_data.pxn_dt == target_date4
mask = mask1 | mask2 | mask3 | mask4
test_data = test_data.loc[mask]



"Step 3: Data preprocessing"

def advanced_preprocessing(test_data):


  test_data['Industry'].fillna(test_data['Industry'].mode()[0], inplace=True)
  # feature engineering

  test_data['Income_to_Loan_Ratio'] = test_data['FINAL_VALIDATED_INCOME'] / test_data['Loan.Amount']
  test_data['Loan_to_Price_Ratio'] = test_data['Loan.Amount'] / test_data['SellingPrice']
  # test_data['Interest_Rate'] = (test_data['Loan.Amount'] - test_data['DownPayment']) / (test_data['Amortization'] * 12)
  test_data['Loan_Tenure_Years'] = test_data['Term'] / 12
  test_data['Monthly_Debt'] = test_data['Amortization']
  test_data['DTI'] = (test_data['Monthly_Debt'] / data['FINAL_VALIDATED_INCOME'])*100
  test_data['Down_Loan'] = test_data['DownPayment']/ test_data['Loan.Amount']
  test_data['Down_SP'] = test_data['DownPayment']/ test_data['SellingPrice']

  # one-hot encoding

  test_data= pd.get_dummies(test_data,
  columns=[ 'NATIONALITY', 'CivilStatus', 'Gender', 'ID.Type', 'Dealer', 'Brand','MC.Unit', 'Type.of.Recidency', 'Education', 'Address', 'Source.of.Income',
        'Classification', 'Industry.Classification.2', 'Industry' ])
  # Scaling

  numerical_columns = ['FINAL_VALIDATED_INCOME', 'DownPayment', 'Loan.Amount',
                      'SellingPrice', 'Term', 'Amortization','Income_to_Loan_Ratio','Loan_to_Price_Ratio', 'Loan_Tenure_Years', 'Monthly_Debt', 'DTI', 'cc', 'Down_Loan', 'Down_SP']
  scaler = PowerTransformer(method='yeo-johnson', standardize=False)  # 'box-cox' or 'yeo-johnson'
  test_data[numerical_columns] = scaler.fit_transform(test_data[numerical_columns])
  # test_data[numerical_columns] = scaler.transform(test_data[numerical_columns])
  return test_data



target = test_data.DPD90
test_data.drop(columns = ['pxn_dt', 'DPD90'], inplace = True)

test_data = advanced_preprocessing(test_data)
feature_names = test_data.columns


"Step 4 : Feature importance calculation"


# Create a RandomForestClassifier for feature seection
classifier = RandomForestClassifier(n_estimators=100, random_state=42)

# Train the classifier

classifier.fit(test_data, target)

# Extract feature importances

feature_importances = classifier.feature_importances_
indices = np.argsort(feature_importances)[::-1]
sorted_feature_names = [feature_names[i] for i in indices]
sorted_importances = [feature_importances[i] for i in indices]

feature_names = list(test_data.columns)
# Create a DataFrame to display feature names and their importances
feature_importance_df = pd.DataFrame({
    'Feature': feature_names,
    'Importance': feature_importances
})

# Sort features by importance
feature_importance_df = feature_importance_df.sort_values(by='Importance', ascending=False)




top_25_features =  feature_importance_df.sort_values(by='Importance', ascending=False).head(25)
plt.figure(figsize=(12, 8))
plt.barh(top_25_features['Feature'], top_25_features['Importance'], color='skyblue')
plt.xlabel('Importance')
plt.ylabel('Feature')
plt.title('Top 25 Features and Their Importance')
plt.gca().invert_yaxis()  # Invert y-axis for better readability
plt.show()


feature_data = test_data.loc[:, ['ACCOUNT_NO', "FINAL_VALIDATED_INCOME","DTI", "Income_to_Loan_Ratio", "Age","Amortization", "Loan.Amount", "Monthly_Debt", "DownPayment",
                                 "Loan_to_Price_Ratio", "Down_Loan", "Down_SP", "SellingPrice", "ID.Type_BIR", "cc",
                                  "Education_COLLEGE GRADUATE", "CivilStatus_SINGLE", "Education_COLLEGE UNDERGRADUATE",
                                  "Term", "Type.of.Recidency_LIVING WITH PARENT","Loan_Tenure_Years","CivilStatus_MARRIED", "ID.Type_LTO",
                                  "Type.of.Recidency_OWNED", "Education_HIGH SCHOOL GRADUATE", "CivilStatus_LIVE-IN", "Gender_FEMALE", "Gender_MALE","Industry.Classification.2_Others",
                                  "Industry_OTHERS", "ID.Type_BARANGAY HALL","Source.of.Income_PRIVATE EMPLOYEE","Dealer_CICLO SUERTE", "Brand_HONDA",
                                 "Dealer_MAVERICK", "Source.of.Income_BUSINESS", "Industry_DELIVERY RIDER", "Brand_YAMAHA", "ID.Type_SSS", "ID.Type_PSA",
                                 "Industry.Classification.2_Transportation and Storage", "Type.of.Recidency_RENTED", "Industry.Classification.2_Wholesale and retail trade; repair of motor vehicles and motorcycles",
                                  "Industry.Classification.2_Administrative and support service activities", "Dealer_KSERVICO", "Dealer_CELEBES", "Address_SOUTH COTABATO, GENERAL SANTOS CITY (DADIANGAS), LABANGAL",
                                  "Type.of.Recidency_LIVING WITH RELATIVE", "ID.Type_COMELEC", "ID.Type_PHILHEALTH", "Dealer_RIZAL AUTOZONE", "Source.of.Income_SELF-EMPLOYED",
                                 "Industry_SARI-SARI STORE OWNER", "Source.of.Income_GOVERNMENT EMPLOYEE", "Dealer_MOTORCYCLE CITY", "Industry.Classification.2_Financial and Isurance Activities",
                                 "Type.of.Recidency_STAY-IN", "Classification_REGULAR BIKE", "Brand_KAWASAKI", "Brand_SUZUKI", "MC.Unit_PCX160 (ABS VARIANT)", "Industry_Remittance",
                                 "Classification_HIGH-END BIKE"]]


"Step 5 : Model trainig"

X_train, X_test, y_train, y_test = train_test_split(feature_data, target, test_size=0.2, random_state=42)
params = {
    'eta':  0.1,                      # Learning rate (shrinkage)
    'alpha': 0.01 ,                    # L1 regularization (Lasso)
    'lambda': 0.01,                   # L2 regularization (Ridge)



    }

xgb_model = xgb.XGBClassifier(objective='binary:logistic', random_state=42)
xgb_model.fit(X_train, y_train)



V_data = data.copy()
V_data.drop(columns = ['APPLICATION.NO','CUSTOMER_NO', 'DPD', "Customer.Number"], inplace = True)
target_date1 = pd.to_datetime('2023-01')
target_date2 = pd.to_datetime('2022-12')
# target_date3 = pd.to_datetime('2023-02')

mask1 = V_data.pxn_dt == target_date1
mask2 = V_data.pxn_dt == target_date2
# mask3 = V_data.pxn_dt == target_date3
mask = mask1 | mask2
V_data = V_data.loc[mask]



V_target = V_data.DPD90
V_data.drop(columns = ['pxn_dt', 'DPD90'], inplace = True)
V_data = advanced_preprocessing(V_data)
feature_names = V_data.columns



V_feature_data = V_data.loc[:, ['ACCOUNT_NO', "FINAL_VALIDATED_INCOME","DTI", "Income_to_Loan_Ratio", "Age","Amortization", "Loan.Amount", "Monthly_Debt", "DownPayment",
                                 "Loan_to_Price_Ratio", "Down_Loan", "Down_SP", "SellingPrice", "ID.Type_BIR", "cc",
                                  "Education_COLLEGE GRADUATE", "CivilStatus_SINGLE", "Education_COLLEGE UNDERGRADUATE",
                                  "Term", "Type.of.Recidency_LIVING WITH PARENT","Loan_Tenure_Years","CivilStatus_MARRIED", "ID.Type_LTO",
                                  "Type.of.Recidency_OWNED", "Education_HIGH SCHOOL GRADUATE", "CivilStatus_LIVE-IN", "Gender_FEMALE", "Gender_MALE","Industry.Classification.2_Others",
                                  "Industry_OTHERS", "ID.Type_BARANGAY HALL","Source.of.Income_PRIVATE EMPLOYEE","Dealer_CICLO SUERTE", "Brand_HONDA",
                                 "Dealer_MAVERICK", "Source.of.Income_BUSINESS", "Industry_DELIVERY RIDER", "Brand_YAMAHA", "ID.Type_SSS", "ID.Type_PSA",
                                 "Industry.Classification.2_Transportation and Storage", "Type.of.Recidency_RENTED", "Industry.Classification.2_Wholesale and retail trade; repair of motor vehicles and motorcycles",
                                  "Industry.Classification.2_Administrative and support service activities", "Dealer_KSERVICO", "Dealer_CELEBES", "Address_SOUTH COTABATO, GENERAL SANTOS CITY (DADIANGAS), LABANGAL",
                                  "Type.of.Recidency_LIVING WITH RELATIVE", "ID.Type_COMELEC", "ID.Type_PHILHEALTH", "Dealer_RIZAL AUTOZONE", "Source.of.Income_SELF-EMPLOYED",
                                 "Industry_SARI-SARI STORE OWNER", "Source.of.Income_GOVERNMENT EMPLOYEE", "Dealer_MOTORCYCLE CITY", "Industry.Classification.2_Financial and Isurance Activities",
                                 "Type.of.Recidency_STAY-IN", "Classification_REGULAR BIKE", "Brand_KAWASAKI", "Brand_SUZUKI", "MC.Unit_PCX160 (ABS VARIANT)", "Industry_Remittance",
                                 "Classification_HIGH-END BIKE"]]



"Step 6 : Plotting the validation curves"


def score(expected,predicted, predicted_prob,cutoff):
    target_0=len([x for x in expected if x==0])
    target_1=len([x for x in expected if x==1])
    bad_rate=target_1/(target_1+target_0)

    fpr, tpr, threshold_train = roc_curve(expected,predicted_prob)
    roc_auc = auc(fpr, tpr)
    gini = (roc_auc*2-1)
    predicted_bool=[1 if x>=cutoff else 0 for x in predicted]
    Accuracy=accuracy_score(expected,predicted_bool)
    Precision=precision_score(expected,predicted_bool)
    Sensitivity=recall_score(expected,predicted_bool)
    Specificity=recall_score(expected,predicted_bool, pos_label=0)
    F1=f1_score(expected,predicted_bool)
    Balanced_accuracy=balanced_accuracy_score(expected,predicted_bool)
    cm=confusion_matrix(expected,predicted_bool)
    TP=cm[1][1]
    FP=cm[0][1]
    TN=cm[0][0]
    FN=cm[1][0]
    metrics=[target_0,target_1,bad_rate,roc_auc,gini,cutoff,Accuracy,Balanced_accuracy,Sensitivity,Specificity,Precision,F1,TP,FP,FN,TN]
    return metrics

# Predict on the test set
y_pred_test = xgb_model.predict(X_test)
y_pred_prob_test = xgb_model.predict_proba(X_test)[:, 1]
# Predit on the VaI data
y_pred_v = xgb_model.predict(V_feature_data)
y_pred_prob_v = xgb_model.predict_proba(V_feature_data)[:, 1]
# Predict on train
y_pred_train = xgb_model.predict(X_train)
y_pred_prob_train = xgb_model.predict_proba(X_train)[:, 1]
# Predict on train data
cutoff=0.19
index=['Majority_obs','Minority_obs','Bad_rate','AUC','Gini','Cut_off','Accuracy','Balanced_Accuracy','Sensitivty','Specificity','Precision','F1',
         'TP','FP','FN','TN']
score_test=score(y_test,y_pred_test, y_pred_prob_test, cutoff)
score_train=score(y_train,y_pred_train,y_pred_prob_train, cutoff)

score_oot=score(V_target, y_pred_v, y_pred_prob_v,cutoff)

scores=np.array([score_train,score_test,score_oot])
pd.DataFrame(np.transpose(scores), columns = ['Train' , 'Test' ,'OOT'], index=index)
