import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import cross_val_score
from sklearn import tree
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import confusion_matrix, precision_score, recall_score, f1_score
from imblearn.under_sampling import RandomUnderSampler
from imblearn.over_sampling import RandomOverSampler
from sklearn.metrics import zero_one_loss
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import confusion_matrix
from imblearn.over_sampling import SMOTE
from sklearn.metrics import roc_auc_score
from sklearn.metrics import mean_squared_error, r2_score
from tabulate import tabulate

np.random.seed(1234)

df = pd.read_csv("data2.csv")
# Adjust display options for pandas
pd.options.display.max_columns = None
pd.options.display.max_rows = None
pd.options.display.float_format = '{:.2f}'.format

# Generate the summary table
summary_table = df.describe()

# Convert the summary table to LaTeX format
latex_table = tabulate(summary_table, headers='keys', tablefmt='latex')

# Print the LaTeX table
print(latex_table)

df['duration'] = df['duration'].astype(float)
df['property_magnitude'] = df['property_magnitude'].astype(str)
df['checking_status'] = df['checking_status'].astype(str)
df['credit_history'] = df['credit_history'].astype(str)
df['purpose'] = df['purpose'].astype(str)
df['credit_amount'] = df['credit_amount'].astype(float)
df['savings_status'] = df['savings_status'].astype(str)
df['employment'] = df['employment'].astype(str)
df['installment_commitment'] = df['installment_commitment'].astype(float)
df['personal_status'] = df['personal_status'].astype(str)
df['other_parties'] = df['other_parties'].astype(str)
df['residence_since'] = df['residence_since'].astype(float)
df['age'] = df['age'].astype(float)
df['other_payment_plans'] = df['other_payment_plans'].astype(str)
df['housing'] = df['housing'].astype(str)
df['existing_credits'] = df['existing_credits'].astype(float)
df['job'] = df['job'].astype(str)
df['num_dependents'] = df['num_dependents'].astype(float)
df['own_telephone'] = df['own_telephone'].astype(str)
df['foreign_worker'] = df['foreign_worker'].astype(str)
df['class'] = df['class'].astype(str)


# Example usage
X = df.drop(['class'], axis=1)  # Input features
y = df['class'].values  # True labels

# Initialize and train the classifier (import cross_val_score and DecisionTreeClassifier)
clf = DecisionTreeClassifier(random_state=0)
xdepthlist = []
cvlist = []
tree_depth = range(1, 30)
for d in tree_depth:
    xdepthlist.append(d)
    clf.max_depth = d
    cv = np.mean(cross_val_score(clf, X, y, cv=10, scoring='accuracy'))
    cvlist.append(cv)


# Print the depth and corresponding cross-validation score
# for depth, score in zip(xdepthlist, cvlist):
#    print(f"Depth: {depth}, Cross-Validation Score: {score}")


plt.figure(figsize=(15, 10))
plt.xlabel('Tree Depth', fontsize=18, color='black')
plt.ylabel('Cross-Validation score', fontsize=18, color='black')
plt.plot(xdepthlist, cvlist, '-*', linewidth=0.5)
plt.show()

'''
Initially. as the depth increases, the model becomes more complex and captures more patterns. 
This could lead to an improved performance.
After a certain point increasing depth might cause overfitting. 
If a model overfits the data, it is too specific to the training data and fails the perform well on new data. 
We observe that the cross-validation scores reach their max around depth 3 and 4. 
Further increasing the depth does not yield better validation scores. 
'''

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=None)

# Data balancing using undersampling
# rus = RandomUnderSampler(random_state=0)
# X_train, y_train = rus.fit_resample(X_train, y_train)

# Data balancing using oversampling
# ros = RandomOverSampler(random_state=0)
# X_train, y_train = ros.fit_resample(X_train, y_train)

# Data balancing using SMOTE
smote = SMOTE()
X_train, y_train = smote.fit_resample(X_train, y_train)


#Decision tree\\
new_depth = 3
clf_decisiontree = DecisionTreeClassifier(random_state=None, max_depth=new_depth)
model = clf_decisiontree.fit(X_train, y_train)
y_predict = clf_decisiontree.predict(X_test)
confusion = confusion_matrix(y_test, y_predict)
tree_accuracy = np.trace(confusion) / np.sum(confusion)
tn, fp, fn, tp = confusion_matrix(y_test, y_predict).ravel()
tree_specificity = tn / (tn+fp)

# Calculate precision, recall, and F1-score. Pos_label 1 (good)
precision_DT = precision_score(y_test, y_predict, pos_label='2')
recall_DT = recall_score(y_test, y_predict, pos_label='2')
f1_DT = f1_score(y_test, y_predict, pos_label='2')
predicted_probabilities = model.predict_proba(X_test)[:, 1]
true_labels = y_test
auc_roc_DT = roc_auc_score(true_labels, predicted_probabilities)

'''
fig = plt.figure(figsize=(25, 20))
fig_tree = tree.plot_tree(clf_decisiontree,
                          feature_names=['checking_status', 'credit_history', 'purpose', 'credit_amount',
                                         'savings_status', 'employment', 'installment_commitment', 'personal_status',
                                         'other_parties', 'residence_since', 'age', 'other_payment_plans', 'housing',
                                         'existing_credits', 'job', 'num_dependents', 'own_telephone',
                                         'foreign_worker'],
                          class_names=['good', 'bad'],
                          filled=True)
plt.xlabel('Decision Tree', fontsize=18, color='black')
plt.ylabel('Test', fontsize=18, color='black')
plt.show()
'''
importance = model.feature_importances_
indices = np.argsort(importance)[::-1]

if np.sum(importance) == 0:
    print("Features are of no importance.")
else:
    print("The Feature Importance is:")
    for f in range(X_train.shape[1]):
        print("Feature %d (%f)" % (indices[f], importance[indices[f]]))

importance = importance[indices][::-1]
indices = indices[::-1]

plt.figure(figsize=(10, 6))
plt.barh(range(X_train.shape[1]), importance)
plt.yticks(range(X_train.shape[1]), X_train.columns[indices])
plt.xlabel('Feature Importance')
plt.ylabel('Features')
plt.title('Feature Importance Decision Tree')
plt.tight_layout()
plt.show()



# RANDOM FOREST MODEL
RF_model = RandomForestClassifier(n_estimators=100, max_features="sqrt", random_state=0)
RF_model.fit(X_train, y_train)
RF_model.score(X_train, y_train)
y_predict = RF_model.predict(X_test)

# Calculate scores
RF_confusion = confusion_matrix(y_test, y_predict)
RF_accuracy = np.trace(RF_confusion) / np.sum(RF_confusion)
tn, fp, fn, tp = confusion_matrix(y_test, y_predict).ravel()
RF_specificity = tn / (tn+fp)
precision_RF = precision_score(y_test, y_predict, pos_label='2')
recall_RF = recall_score(y_test, y_predict, pos_label='2')
f1_RF = f1_score(y_test, y_predict, pos_label='2')
predicted_probabilities = RF_model.predict_proba(X_test)[:, 1]
true_labels = y_test
auc_roc_RF = roc_auc_score(true_labels, predicted_probabilities)


print("Accuracy Decision tree:", tree_accuracy, "Accuracy Random Forest:", RF_accuracy)
print("Precision Decision tree:", precision_DT, "Precision Random Forest:", precision_RF)
print("Recall Decision Tree:", recall_DT, "Recall Random Forest:", recall_RF)
print("F1-Score Decision Tree:", f1_DT, "Recall Random Forest:", f1_RF)
print("Specificity Decision Tree:", tree_specificity, "Specificity Random Forest", RF_specificity)
print("AUC-ROC Decision Tree:", auc_roc_DT, "AUC-ROC Random Forest", auc_roc_RF)


importance = RF_model.feature_importances_
indices = np.argsort(importance)[::-1]

if np.sum(importance) == 0:
    print("Features are of no importance.")
else:
    print("The Feature Importance is:")
    for f in range(X_train.shape[1]):
        print("Feature %d (%f)" % (indices[f], importance[indices[f]]))

importance = importance[indices][::-1]
indices = indices[::-1]

plt.figure(figsize=(10, 6))
plt.barh(range(X_train.shape[1]), importance)
plt.yticks(range(X_train.shape[1]), X_train.columns[indices])
plt.xlabel('Feature Importance')
plt.ylabel('Features')
plt.title('Feature Importance Random Forest')
plt.tight_layout()
plt.show()

# confusion = confusion_matrix(y_test, y_predict)
# print("Confusion Matrix:")
# print(confusion)



# Calculate Mean Squared Error (MSE)
# mse = mean_squared_error(y_test, y_predict)
# print("Mean Squared Error:", mse)


# loss = zero_one_loss(y_test, y_predict)
# print(loss)
