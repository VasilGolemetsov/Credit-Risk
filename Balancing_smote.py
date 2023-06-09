import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import cross_val_score
from sklearn import tree
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import confusion_matrix, precision_score, recall_score, f1_score
from sklearn.metrics import zero_one_loss
from sklearn.tree import DecisionTreeClassifier
from imblearn.over_sampling import RandomOverSampler  # Instead of random oversampling we use Smote
from sklearn.metrics import roc_auc_score
from imblearn.over_sampling import SMOTE


'''
Note that in this file we use balancing (Smote) for our dataset.
We see that the precision increases and F1-score increases.
Recall decreases for the Decision Tree, while it stays similar for Random Forest.
'''

df = pd.read_csv("data2.csv")
df.drop(["duration", "property_magnitude"], inplace=True, axis=1)

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

print(type(df['personal_status']))

'''
Data Balancing
df_undersample = df.groupby(['class'])
df_undersample = df_undersample.apply(lambda x: x.sample(df_undersample.size().min(), random_state=0).reset_index(drop=True))
df_undersample = df_undersample.droplevel(['class'])
df_undersample.groupby(['class']).size()
df.groupby(['class']).size()
'''

X = df.drop(['class'], axis=1)  # Input features
y = df['class'].values  # True labels

smote = SMOTE()
X_resampled, y_resampled = smote.fit_resample(X, y)

# Initialize and train the classifier using the resampled data
clf = DecisionTreeClassifier(random_state=0)
xdepthlist = []
cvlist = []
tree_depth = range(1, 30)
for d in tree_depth:
    xdepthlist.append(d)
    clf.max_depth = d
    cv = np.mean(cross_val_score(clf, X_resampled, y_resampled, cv=10, scoring='accuracy'))
    cvlist.append(cv)

# Print the depth and corresponding cross-validation score
for depth, score in zip(xdepthlist, cvlist):
    print(f"Depth: {depth}, Cross-Validation Score: {score}")


plt.figure(figsize=(15, 10))
plt.xlabel('Tree Depth', fontsize=18, color='black')
plt.ylabel('Cross-Validation score', fontsize=18, color='black')
plt.plot(xdepthlist, cvlist, '-*', linewidth=0.5)
plt.show()

'''
Initially, as the depth increases, the model becomes more complex and captures more patterns. 
This could lead to improved performance.
After a certain point, increasing depth might cause overfitting. 
If a model overfits the data, it is too specific to the training data and fails to perform well on new data. 
We observe that the cross-validation scores reach their max around depth 3 and 4. 
Further increasing the depth does not yield better validation scores. 
'''

# Split the resampled data into train and test sets
X_train, X_test, y_train, y_test = train_test_split(X_resampled, y_resampled, test_size=0.25, random_state=None)


#Decision tree
new_depth = 13
clf_decisiontree = DecisionTreeClassifier(random_state=None, max_depth=new_depth)
model_under = clf_decisiontree.fit(X_train, y_train)
y_predict = clf_decisiontree.predict(X_test)
confusion = confusion_matrix(y_test, y_predict)
tree_accuracy = np.trace(confusion) / np.sum(confusion)

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

# Calculate precision, recall, and F1-score. Pos_label 1 (good)
precision_DT = precision_score(y_test, y_predict, pos_label='1')
recall_DT = recall_score(y_test, y_predict, pos_label='1')
f1_DT = f1_score(y_test, y_predict, pos_label='1')

#Random Forest
RF_model = RandomForestClassifier(n_estimators=100, max_features="sqrt", random_state=0)
RF_model.fit(X_train, y_train)
RF_model.score(X_train, y_train)
y_predict = RF_model.predict(X_test)
RF_confusion = confusion_matrix(y_test, y_predict)
RF_accuracy = np.trace(RF_confusion) / np.sum(RF_confusion)
print(RF_accuracy - tree_accuracy)

# Feature Importance
importance = RF_model.feature_importances_
indices = np.argsort(importance)[::-1]

if np.sum(importance) == 0:
    print("All feature importance values are zero.")
else:
    print("Random Forest Feature Importance:")
    for f in range(X_train.shape[1]):
        print("Feature %d (%f)" % (indices[f], importance[indices[f]]))


confusion = confusion_matrix(y_test, y_predict)
print("Confusion Matrix:")
print(confusion)

# Calculate precision, recall, and F1-score. Pos_label 1 (good)
precision_RF = precision_score(y_test, y_predict, pos_label='1')
recall_RF = recall_score(y_test, y_predict, pos_label='1')
f1_RF = f1_score(y_test, y_predict, pos_label='1')


print("Accuracy Decision tree:", tree_accuracy, "Accuracy Random Forest:", RF_accuracy)
print("Precision Decision tree:", precision_DT, "Precision Random Forest:", precision_RF)
print("Recall Decision Tree:", recall_DT, "Recall Random Forest:", recall_RF)
print("F1-Score Decision Tree:", f1_DT, "Recall Random Forest:", f1_RF)


# Assuming you have already trained your classifier and obtained predicted probabilities and true labels
predicted_probabilities = RF_model.predict_proba(X_test)[:, 1]  # Replace 'model' with your trained classifier
true_labels = y_test  # Replace 'y_test' with your true labels

auc_roc = roc_auc_score(true_labels, predicted_probabilities)
print("AUC-ROC:", auc_roc)
