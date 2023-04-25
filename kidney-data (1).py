#!/usr/bin/env python
# coding: utf-8

# In[10]:


import pandas as pd
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report, roc_curve, auc
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
import seaborn as sns


# In[11]:


df=pd.read_csv("/kaggle/input/kidney-stone-dataset/data.csv")


# In[12]:


df


# In[13]:


import seaborn as sns
fig, axes = plt.subplots(nrows=2, ncols=4, figsize=(12, 8))
for i, column in enumerate(df.columns):
    ax = axes[i // 4, i % 4]
    sns.histplot(df[column], ax=ax)
    ax.set_title(column)
plt.tight_layout()
plt.show()


# In[14]:


df.info()


# In[15]:


X = df.drop('target', axis=1)
y = df['target']


# In[16]:


# perform PCA with n_components=2
pca = PCA(n_components=2)
X = pca.fit_transform(X)



sns.scatterplot(x=X[:,0], y=X[:,1], hue=y)
plt.show()


# In[17]:


X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)


# In[18]:


# Model 1: SVM Classifier
svm = SVC(probability=True)
svm.fit(X_train, y_train)
y_pred_svm = svm.predict(X_test)
print(' Default SVM Classifier Accuracy:', accuracy_score(y_test, y_pred_svm))
print(' Default SVM Classifier Report:\n', classification_report(y_test, y_pred_svm))


# In[19]:


#Hyper Parameter tuning for SVM Classifier

param_grid = {'C': [ 0.1,1], 'kernel': ['linear', 'poly', 'rbf', 'sigmoid'], }
svm_grid = GridSearchCV(svm, param_grid, cv=5)
svm_grid.fit(X_train, y_train)
print("Best hyperparameters for SVM Classifier: ", svm_grid.best_params_)
svm_tuned = SVC(C=svm_grid.best_params_['C'], kernel=svm_grid.best_params_['kernel'], random_state=42, probability=True)
svm_tuned.fit(X_train, y_train)
y_pred_svm_tuned = svm_tuned.predict(X_test)
print('Tuned SVM Accuracy:', accuracy_score(y_test, y_pred_svm_tuned))
print('Tuned SVM Classifier Report:\n', classification_report(y_test, y_pred_svm_tuned))


# In[20]:


# Model 2: Decision Tree
dt = DecisionTreeClassifier()
dt.fit(X_train, y_train)
y_pred_dt = dt.predict(X_test)
print(' Default Decision Tree Accuracy:', accuracy_score(y_test, y_pred_dt))
print(' Default Decision Tree Report:\n', classification_report(y_test, y_pred_dt))


# In[21]:


#Hyper Parameter tuning for Decision Tree
dt_hyperparameters = {
    'max_depth': [3, 5, 7, 9],
    'min_samples_split': [2, 4, 6, 8, 10]
}
grid_dt = GridSearchCV(dt, dt_hyperparameters, cv=5)
grid_dt.fit(X_train, y_train)
print("Best hyperparameters for Decision Tree Classifier: ", grid_dt.best_params_)
dt_tuned = DecisionTreeClassifier(max_depth=grid_dt.best_params_['max_depth'], min_samples_split=grid_dt.best_params_['min_samples_split'], random_state=42)
dt_tuned.fit(X_train, y_train)
y_pred_dt_tuned = dt_tuned.predict(X_test)
print('Tuned Decision Tree Classifier Accuracy:', accuracy_score(y_test, y_pred_dt_tuned))
print('Tuned Decision Tree Classifier Report:\n', classification_report(y_test, y_pred_dt_tuned))


# In[22]:


# Model 3: Random Forest
rf = RandomForestClassifier()
rf.fit(X_train, y_train)
y_pred_rf = rf.predict(X_test)
print('Random Forest Default Accuracy:', accuracy_score(y_test, y_pred_rf))
print(' Random Forest Default Report:\n', classification_report(y_test, y_pred_rf))


# In[23]:


#Hyper Parameter tuning for Random Forest
rf_param_grid = {'n_estimators': [50, 100, 150], 'max_depth': [2, 4, 6], 'min_samples_split': [2, 4, 6]}
rf_grid = GridSearchCV(estimator=rf, param_grid=rf_param_grid, cv=5)
rf_grid.fit(X_train, y_train)
print('Best Hyperparameters for Random Forest Classifier:', rf_grid.best_params_)
rf_tuned = RandomForestClassifier(n_estimators=rf_grid.best_params_['n_estimators'],max_depth=rf_grid.best_params_['max_depth'], min_samples_split=rf_grid.best_params_['min_samples_split'], random_state=42)
rf_tuned.fit(X_train, y_train)
y_pred_rf_tuned = dt_tuned.predict(X_test)
print('Tuned Decision Tree Classifier Accuracy:', accuracy_score(y_test, y_pred_rf_tuned))
print('Tuned Decision Tree Classifier Report:\n', classification_report(y_test, y_pred_rf_tuned))


# In[24]:


# Generate ROC curves and AUC scores for each model
fpr_svm, tpr_svm, thresholds_svm = roc_curve(y_test, svm.predict_proba(X_test)[:,1])
roc_auc_svm = auc(fpr_svm, tpr_svm)

fpr_svm_tuned, tpr_svm_tuned, thresholds_svm_tuned = roc_curve(y_test, svm_tuned.predict_proba(X_test)[:,1])
roc_auc_svm_tuned = auc(fpr_svm_tuned, tpr_svm_tuned)

fpr_dt, tpr_dt, thresholds_dt = roc_curve(y_test, dt.predict_proba(X_test)[:,1])
roc_auc_dt = auc(fpr_dt, tpr_dt)

fpr_dt_tuned, tpr_dt_tuned, thresholds_dt_tuned = roc_curve(y_test, dt_tuned.predict_proba(X_test)[:,1])
roc_auc_dt_tuned = auc(fpr_dt_tuned, tpr_dt_tuned)

fpr_rf, tpr_rf, thresholds_rf = roc_curve(y_test, rf.predict_proba(X_test)[:,1])
roc_auc_rf = auc(fpr_rf, tpr_rf)

fpr_rf_tuned, tpr_rf_tuned, thresholds_rf_tuned = roc_curve(y_test, rf_tuned.predict_proba(X_test)[:,1])
roc_auc_rf_tuned = auc(fpr_rf_tuned, tpr_rf_tuned)

plt.figure(figsize=(6,6))
plt.title('Receiver Operating Characteristic (ROC) Curve')
plt.plot(fpr_svm, tpr_svm, 'b', label='SVM Classifier AUC = %0.2f' % roc_auc_svm)
plt.plot(fpr_svm_tuned, tpr_svm_tuned, 'b', label='SVM Tuned Classifier AUC = %0.2f' % roc_auc_svm_tuned)
plt.plot(fpr_dt, tpr_dt, 'g', label='Decision Tree AUC = %0.2f' % roc_auc_dt)
plt.plot(fpr_dt_tuned, tpr_dt_tuned, 'g', label='Decision Tree Tuned AUC = %0.2f' % roc_auc_dt_tuned)
plt.plot(fpr_rf, tpr_rf, 'r', label='Random Forest AUC = %0.2f' % roc_auc_rf)
plt.plot(fpr_rf_tuned, tpr_rf_tuned, 'r', label='Random Forest Tuned AUC = %0.2f' % roc_auc_rf_tuned)
plt.legend(loc='lower right')
plt.plot([0, 1], [0, 1],'r--')
plt.xlim([0, 1])
plt.ylim([0, 1])
plt.ylabel('True Positive Rate')
plt.xlabel('False Positive Rate')
plt.show()


# In[25]:


# Generate confusion matrices for each model
cm_svm = confusion_matrix(y_test, y_pred_svm)
cm_svm_tuned = confusion_matrix(y_test, y_pred_svm_tuned)
cm_dt = confusion_matrix(y_test, y_pred_dt)
cm_dt_tuned = confusion_matrix(y_test, y_pred_dt_tuned)
cm_rf = confusion_matrix(y_test, y_pred_rf)
cm_rf_tuned = confusion_matrix(y_test, y_pred_rf_tuned)

plt.figure(figsize=(8,8))
plt.suptitle('Confusion Matrices')

plt.subplot(2,3,1)
sns.heatmap(cm_svm, annot=True, fmt='d', cmap='Blues')
plt.title('SVM Classifier')

plt.subplot(2,3,2)
sns.heatmap(cm_svm_tuned, annot=True, fmt='d', cmap='Blues')
plt.title('SVM  Tuned Classifier')

plt.subplot(2,3,3)
sns.heatmap(cm_dt, annot=True, fmt='d', cmap='Greens')
plt.title('Decision Tree')

plt.subplot(2,3,4)
sns.heatmap(cm_dt_tuned, annot=True, fmt='d', cmap='Greens')
plt.title('Decision Tree Tuned')

plt.subplot(2,3,5)
sns.heatmap(cm_rf, annot=True, fmt='d', cmap='Reds')
plt.title('Random Forest')

plt.subplot(2,3,6)
sns.heatmap(cm_rf_tuned, annot=True, fmt='d', cmap='Reds')
plt.title('Random Forest Tuned')

plt.show()

