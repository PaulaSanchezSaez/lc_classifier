import numpy as np
import pandas as pd
from sklearn import preprocessing, model_selection, metrics, ensemble
import pickle
import itertools
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from collections import Counter
from imblearn.ensemble import BalancedRandomForestClassifier as RandomForestClassifier
from scipy.stats import randint as sp_randint

#names of files with detections, features and labels for the training set (v3)
date = '20191119'

labels_file = 'dfcrossmatches_prioritized_v4.csv'

features_file = 'features20191119.pkl'

features_paps = 'paps_features_all_with_mean.pkl'

class_output = 'classification_unlabelled_set_20191119.csv'

model_first_layer = 'stat_prob_hierRF_model_2/rf_model_2_hierarchical_layer_'+date
model_periodic_layer = 'stat_prob_hierRF_model_2/rf_model_2_periodic_layer_'+date
model_transient_layer = 'stat_prob_hierRF_model_2/rf_model_2_transient_layer_'+date
model_stochastic_layer = 'stat_prob_hierRF_model_2/rf_model_2_stochastic_layer_'+date

conf_matrix_name_first_layer = 'stat_prob_hierRF_model_2/confusion_matrix_rf_model_2_hierarchical_layer_'+date
conf_matrix_name_second_layer = 'stat_prob_hierRF_model_2/confusion_matrix_rf_model_2_multiclass_'+date

feature_importance_name_first_layer = 'stat_prob_hierRF_model_2/feature_importance_rf_model_2_hierarchical_layer_'+date
feature_importance_name_periodic_layer = 'stat_prob_hierRF_model_2/feature_importance_rf_model_2_periodic_layer_'+date
feature_importance_name_transient_layer = 'stat_prob_hierRF_model_2/feature_importance_rf_model_2_transient_layer_'+date
feature_importance_name_stochastic_layer = 'stat_prob_hierRF_model_2/feature_importance_rf_model_2_stochastic_layer_'+date
################################################################################

#reading the training set files
df_feat = pd.read_pickle(features_file)
df_labels = pd.read_csv(labels_file,index_col='oid')
'''
df_paps = pd.read_pickle(features_paps)
df_paps = df_paps.set_index('oid')
df_paps.paps_ratio_1=df_paps.paps_ratio_1.astype(float)
df_paps.paps_ratio_2=df_paps.paps_ratio_2.astype(float)
df_paps.paps_high_2=df_paps.paps_high_2.astype(float)
df_paps.paps_high_1=df_paps.paps_high_1.astype(float)
df_paps.paps_low_2=df_paps.paps_low_2.astype(float)
df_paps.paps_low_1=df_paps.paps_low_1.astype(float)


df_paps = df_paps.replace([np.inf, -np.inf], np.nan)
df_paps.replace([-99], -999,inplace=True)
df_paps.fillna(-999,inplace=True)
'''
#discarging infinite values
df_feat = df_feat.replace([np.inf, -np.inf], np.nan)

#creating new labels to combine SNII and SNIIb classes, and to add RS-CVn as a new class
df_labels['class_original'] = df_labels['classALeRCE']
df_labels.loc[(df_labels['class_source'] == 'RS CVn'), 'class_original'] = 'RS-CVn'
df_labels.loc[(df_labels['class_original'] == 'SNIIn'), 'class_original'] = 'SNII'
df_labels.loc[(df_labels['class_original'] == 'AGN-I'), 'class_original'] = 'QSO-I'
df_labels.loc[(df_labels['class_source'] == 'A') | (df_labels['class_source'] == 'AGN_galaxy_dominated'), 'class_original'] = 'AGN-I'


#defining the classes included in the RF model
label_order = ['QSO-I','AGN-I', 'Blazar', 'CV/Nova', 'SNIa', 'SNIbc', 'SNII',
               'SLSN', 'EBSD/D', 'EBC', 'DSCT', 'RRL', 'Ceph', 'LPV','RS-CVn','Periodic-Other']

labels = df_labels.loc[df_labels.class_original.isin(label_order)][["class_original"]]

#defining hierarchical classes:

labels['class_hierachical'] = labels['class_original']

labels.loc[ (labels['class_hierachical'] == 'Periodic-Other') | (labels['class_hierachical'] == 'EBSD/D') | (labels['class_hierachical'] == 'EBC')  | (labels['class_hierachical'] == 'DSCT') | (labels['class_hierachical'] == 'RRL') | (labels['class_hierachical'] == 'Ceph') , 'class_hierachical'] = 'Periodic'

labels.loc[(labels['class_hierachical'] == 'SNIa') | (labels['class_hierachical'] == 'SNIbc') | (labels['class_hierachical'] == 'SNII') | (labels['class_hierachical'] == 'SLSN'), 'class_hierachical'] = 'Transient'

labels.loc[(labels['class_hierachical'] == 'RS-CVn') |  (labels['class_hierachical'] == 'CV/Nova')  |   (labels['class_hierachical'] == 'AGN-I') |  (labels['class_hierachical'] == 'QSO-I') | (labels['class_hierachical'] == 'Blazar')  | (labels['class_hierachical'] == 'LPV') , 'class_hierachical'] = 'Stochastic'

cm_classes_hierachical = ['Stochastic','Transient','Periodic']
cm_classes_original = label_order

#defining columns excluded from the df_nd table

rm_nd_cols = [
'n_det_fid_1',
'n_det_fid_2',
'n_pos_1',
'n_pos_2',
'n_neg_1',
'n_neg_2',
'paps_non_zero_1',
'paps_non_zero_2',
]

#paps_drop = ['paps_non_zero_1','paps_PN_flag_1','paps_non_zero_2','paps_PN_flag_2']
#paps_drop = ['paps_non_zero_1','paps_non_zero_2']

#combining all the DF
df = labels.join(df_feat.drop(rm_nd_cols, axis=1),how='inner')#.join(df_paps.drop(paps_drop, axis=1))
df = df.replace([np.inf, -np.inf], np.nan)
df_train = df.copy()
df_train = df_train.fillna(-999)
df.drop(['Mean_1','Mean_2','class_original','class_hierachical'], axis=1, inplace=True)
df = df.fillna(-999)

labels = labels.loc[df.index.values]


################################################################################
#Defining functions to plot the confusion matrix and the feature importance

def plot_confusion_matrix(cm, classes, plot_name,
                          normalize=True,
                          title='Confusion matrix',
                          cmap=plt.cm.Blues):
    """
    This function prints and plots the confusion matrix.
    Normalization can be applied by setting `normalize=True`.
    """
    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        print("Normalized confusion matrix")
    else:
        print('Confusion matrix, without normalization')



    fig, ax = plt.subplots(figsize=(12, 10))
    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    plt.title(title)
    plt.colorbar()
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes, rotation=45)
    plt.yticks(tick_marks, classes)

    fmt = '.2f' if normalize else 'd'
    thresh = cm.max() / 2.
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        plt.text(j, i, format(cm[i, j], fmt),
                 horizontalalignment="center",
                 color="white" if cm[i, j] > thresh else "black")

    plt.tight_layout()
    plt.ylabel('True label')
    plt.xlabel('Predicted label')
    plt.savefig(plot_name, bbox_inches='tight')
    plt.close()


def plot_feature_importances(model, feature_names,feature_importances_name):
    I = np.argsort(model.feature_importances_)[::-1]
    fig, ax = plt.subplots(figsize=(16, 5), tight_layout=True)
    x_plot = np.arange(len(model.feature_importances_))
    plt.xticks(x_plot, [feature_names[i] for i in I], rotation='vertical')
    ax.bar(x_plot, height=model.feature_importances_[I]);
    plt.savefig(feature_importances_name)

    plt.close()


################################################################################
#Pre-processing training data

Y_hierarchical = labels['class_hierachical']#.values
Y_original = labels['class_original']#.values

X_hierarchical = df#.columns.values.tolist()

#splitting training set
X_train_hierarchical, X_test_hierarchical, y_train_hierarchical, y_test_hierarchical, y_train_original, y_test_original  = model_selection.train_test_split(X_hierarchical,
Y_hierarchical, Y_original, test_size=0.2, stratify=Y_original)


# separating training sets for sub-classes
X_train_periodic = X_train_hierarchical.loc[y_train_hierarchical=='Periodic', :]
y_train_periodic = y_train_original.loc[y_train_hierarchical=='Periodic']

X_train_stochastic = X_train_hierarchical.loc[y_train_hierarchical=='Stochastic', :]
y_train_stochastic = y_train_original.loc[y_train_hierarchical=='Stochastic']

X_train_transient = X_train_hierarchical.loc[y_train_hierarchical=='Transient', :]
y_train_transient = y_train_original.loc[y_train_hierarchical=='Transient']


X_test_periodic = X_test_hierarchical#.drop(['Mean_2'], axis=1)
X_test_stochastic = X_test_hierarchical#.drop(['Mean_2'], axis=1)
X_test_transient = X_test_hierarchical#.drop(['Mean_2'], axis=1)


print(len(y_train_periodic), len(y_train_stochastic), len(y_train_transient))

################################################################################
#Balanced random forest
#First layer: separating Periodic, Stochastic and Transients:

#Training first layer of the RF model

rf_model_hierarchical = RandomForestClassifier(
            n_estimators=600,
            max_features='auto',
            max_depth=None,
            n_jobs=1,
            class_weight='balanced_subsample',
            criterion='entropy',
            min_samples_split=2,
            min_samples_leaf=1)

#param_grid = {"n_estimators":[400,600], "max_features": [0.2,0.4,"auto"],"max_depth":[None,5,10,15]}
param_grid = {"max_features": [0.2,0.4,"auto"],"max_depth":[None,5,10,15]}
modeselektor_hierarchical = model_selection.GridSearchCV(estimator=rf_model_hierarchical, param_grid=param_grid, cv=5, n_jobs=-1)

modeselektor_hierarchical.fit(X_train_hierarchical, y_train_hierarchical)
print("Model selected: \"%s\"" % modeselektor_hierarchical.best_estimator_)


#testing first layer performance

y_true, y_pred = y_test_hierarchical, modeselektor_hierarchical.predict(X_test_hierarchical)
y_pred_proba_hier = modeselektor_hierarchical.predict_proba(X_test_hierarchical)

classes_order_proba_hierarchical = modeselektor_hierarchical.classes_
print(classes_order_proba_hierarchical)

print("Accuracy:", metrics.accuracy_score(y_true, y_pred))
print("Balanced accuracy:", metrics.balanced_accuracy_score(y_true, y_pred))

#Dumping trained model

features_hierarchical = list(X_train_hierarchical)

with open(model_first_layer, 'wb') as pickle_file:
        model_dump = {
            'rf_model': modeselektor_hierarchical,#rf_model_hierarchical,
            'features': features_hierarchical,
            'order_classes': classes_order_proba_hierarchical
            }
        pickle.dump(model_dump, pickle_file)

#plotting confusion matrix
cnf_matrix = metrics.confusion_matrix(y_true, y_pred, labels=cm_classes_hierachical)
print(cnf_matrix)
plot_confusion_matrix(cnf_matrix,cm_classes_hierachical,conf_matrix_name_first_layer)

#plotting feature importance
plot_feature_importances(modeselektor_hierarchical.best_estimator_, features_hierarchical, feature_importance_name_first_layer)

################################################################################
#Training Periodic layer

rf_model_periodic = RandomForestClassifier(
            n_estimators=600,
            max_features='auto',
            max_depth=None,
            n_jobs=1,
            class_weight='balanced_subsample',
            criterion='entropy',
            min_samples_split=2,
            min_samples_leaf=1)



modeselektor_periodic = model_selection.GridSearchCV(estimator=rf_model_periodic, param_grid=param_grid, cv=5, n_jobs=-1)

modeselektor_periodic.fit(X_train_periodic, y_train_periodic)
print("Model selected: \"%s\"" % modeselektor_periodic.best_estimator_)

# Applying periodic model to the test data
y_true_periodic, y_pred_periodic = y_test_original, modeselektor_periodic.predict(X_test_periodic)
y_pred_proba_periodic = modeselektor_periodic.predict_proba(X_test_periodic)

classes_order_proba_periodic = modeselektor_periodic.classes_
print(classes_order_proba_periodic)

#Dumping trained model

features_periodic = list(X_train_periodic)

with open(model_periodic_layer, 'wb') as pickle_file:
        model_dump = {
            'rf_model': modeselektor_periodic,
            'features': features_periodic,
            'order_classes': classes_order_proba_periodic
            }
        pickle.dump(model_dump, pickle_file)

#plotting feature importance
print(len(feature_importance_name_first_layer))
plot_feature_importances(modeselektor_periodic.best_estimator_, features_hierarchical, feature_importance_name_first_layer)

################################################################################
#Training Stochastic layer

rf_model_stochastic = RandomForestClassifier(
            n_estimators=600,
            max_features='auto',
            max_depth=None,
            n_jobs=1,
            class_weight='balanced_subsample',
            criterion='entropy',
            min_samples_split=2,
            min_samples_leaf=1)


modeselektor_stochastic = model_selection.GridSearchCV(estimator=rf_model_stochastic, param_grid=param_grid, cv=5, n_jobs=-1)

modeselektor_stochastic.fit(X_train_stochastic, y_train_stochastic)
print("Model selected: \"%s\"" % modeselektor_stochastic.best_estimator_)

# Applying stochastic model to the test data
y_true_stochastic, y_pred_stochastic  = y_test_original, modeselektor_stochastic.predict(X_test_stochastic)
y_pred_proba_stochastic = modeselektor_stochastic.predict_proba(X_test_stochastic)

classes_order_proba_stochastic = modeselektor_stochastic.classes_
print(classes_order_proba_stochastic)

#Dumping trained model

features_stochastic = list(X_train_stochastic)

with open(model_stochastic_layer, 'wb') as pickle_file:
        model_dump = {
            'rf_model': modeselektor_stochastic,
            'features': features_stochastic,
            'order_classes': classes_order_proba_stochastic
            }
        pickle.dump(model_dump, pickle_file)

#plotting feature importance
plot_feature_importances(modeselektor_stochastic.best_estimator_, features_stochastic, feature_importance_name_stochastic_layer)

################################################################################
#Training Transient layer

rf_model_transient = RandomForestClassifier(
            n_estimators=600,
            max_features='auto',
            max_depth=None,
            n_jobs=1,
            class_weight='balanced_subsample',
            criterion='entropy',
            min_samples_split=2,
            min_samples_leaf=1)


modeselektor_transient = model_selection.GridSearchCV(estimator=rf_model_transient, param_grid=param_grid, cv=5, n_jobs=-1)

modeselektor_transient.fit(X_train_transient, y_train_transient)
print("Model selected: \"%s\"" % modeselektor_transient.best_estimator_)

# Applying transient model to the test data
y_true_transient, y_pred_transient  = y_test_original, modeselektor_transient.predict(X_test_transient)
y_pred_proba_transient = modeselektor_transient.predict_proba(X_test_transient)

classes_order_proba_transient = modeselektor_transient.classes_
print(classes_order_proba_transient)

#Dumping trained model

features_transient = list(X_train_transient)

with open(model_transient_layer, 'wb') as pickle_file:
        model_dump = {
            'rf_model': modeselektor_transient,
            'features': features_transient,
            'order_classes': classes_order_proba_transient
            }
        pickle.dump(model_dump, pickle_file)

#plotting feature importance
plot_feature_importances(modeselektor_transient.best_estimator_, features_transient, feature_importance_name_transient_layer)


################################################################################
#Putting al layers together
# generating final probabilities

#multiplying probabilities of the hierarchical layer with the other layers
prob_periodic = y_pred_proba_periodic*y_pred_proba_hier[:,np.where(classes_order_proba_hierarchical=='Periodic')[0][0]].T[:, np.newaxis]
prob_stochastic = y_pred_proba_stochastic*y_pred_proba_hier[:,np.where(classes_order_proba_hierarchical=='Stochastic')[0][0]].T[:, np.newaxis]
prob_trainsient = y_pred_proba_transient*y_pred_proba_hier[:,np.where(classes_order_proba_hierarchical=='Transient')[0][0]].T[:, np.newaxis]

#obtaining final probabilities matrix
prob_final = np.concatenate((prob_stochastic,prob_trainsient,prob_periodic),axis=1)

print(np.sum(prob_final,axis=1),np.mean(np.sum(prob_final,axis=1)),np.std(np.sum(prob_final,axis=1)))

#getting the ordered name of classes for prob_final
prob_final_class_names = np.concatenate((classes_order_proba_stochastic,classes_order_proba_transient,classes_order_proba_periodic))
print(prob_final_class_names)


class_final_proba = np.amax(prob_final,axis=1)
class_final_index = np.argmax(prob_final,axis=1)
class_final_name = [prob_final_class_names[x] for x in class_final_index]

# generating confusion matrix for multilabels
cnf_matrix = metrics.confusion_matrix(y_test_original, class_final_name,labels=label_order)
print(cnf_matrix)
plot_confusion_matrix(cnf_matrix,label_order, conf_matrix_name_second_layer)

print("Accuracy:", metrics.accuracy_score(y_test_original, class_final_name))
print("Balanced accuracy:", metrics.balanced_accuracy_score(y_test_original, class_final_name))

################################################################################
# Kaggle score

num_y_test =  [np.where(prob_final_class_names==x)[0][0] for x in y_test_original] #label_encoder.transform(y_test)

#print(num_y_test)

CLASSES_REDUCED_V2 = prob_final_class_names

def kaggle_loss(labels, predictions, weights=None):
    np.clip(predictions, 10**-15, 1-10**-15, out=predictions)
    classes = np.unique(labels)
    if weights is None:
        weights = np.ones(len(classes), dtype=np.float64)/len(classes)
    loss_sum = 0
    labels = np.array(labels)
    for i in classes:
        p = predictions[labels == i, i]
        class_score = np.mean(np.log(p))*weights[i]
        print(CLASSES_REDUCED_V2[i], class_score)
        loss_sum += class_score
    return -loss_sum/sum(weights)

print(kaggle_loss(num_y_test,prob_final))

################################################################################
#Classifying unlabeled data

#loading the data

print(df_feat.n_samples_1.size)

rm_nd_cols = ['n_det_fid_1', 'n_det_fid_2', 'n_pos_1', 'n_pos_2', 'n_neg_1', 'n_neg_2',
             'Mean_1','Mean_2','paps_non_zero_1','paps_non_zero_2']

df_feat_ul = df_feat.drop(rm_nd_cols, axis=1)
#df_feat_ul = df_feat_ul.join(df_paps.drop(paps_drop, axis=1))
df_feat_ul.fillna(-999,inplace=True)

print(df_feat_ul.n_samples_1.size)


df_feat_ul_out = df_feat_ul

print(df_feat_ul.n_samples_1.size)

################################################################################
#predicting classes of unlabeled data


test_Y_hierarchical = modeselektor_hierarchical.predict(df_feat_ul)
test_Y_proba_hierarchical = modeselektor_hierarchical.predict_proba(df_feat_ul)

test_Y_periodic = modeselektor_periodic.predict(df_feat_ul)
test_Y_proba_periodic = modeselektor_periodic.predict_proba(df_feat_ul)

test_Y_stochastic = modeselektor_stochastic.predict(df_feat_ul)
test_Y_proba_stochastic = modeselektor_stochastic.predict_proba(df_feat_ul)

test_Y_transient = modeselektor_transient.predict(df_feat_ul)
test_Y_proba_transient = modeselektor_transient.predict_proba(df_feat_ul)


#multiplying probabilities
prob_periodic_ul = test_Y_proba_periodic*test_Y_proba_hierarchical[:,np.where(classes_order_proba_hierarchical=='Periodic')[0][0]].T[:, np.newaxis]
prob_stochastic_ul = test_Y_proba_stochastic*test_Y_proba_hierarchical[:,np.where(classes_order_proba_hierarchical=='Stochastic')[0][0]].T[:, np.newaxis]
prob_trainsient_ul = test_Y_proba_transient*test_Y_proba_hierarchical[:,np.where(classes_order_proba_hierarchical=='Transient')[0][0]].T[:, np.newaxis]

#obtaining final probabilities matrix
prob_final_ul = np.concatenate((prob_stochastic_ul,prob_trainsient_ul,prob_periodic_ul),axis=1)

print(np.sum(prob_final_ul,axis=1),np.mean(np.sum(prob_final_ul,axis=1)),np.std(np.sum(prob_final_ul,axis=1)))

#getting the ordered name of classes for prob_final
prob_final_class_names_ul = np.concatenate((classes_order_proba_stochastic,classes_order_proba_transient,classes_order_proba_periodic))
print(prob_final_class_names)


class_final_proba_ul = np.amax(prob_final_ul,axis=1)
class_final_index_ul = np.argmax(prob_final_ul,axis=1)
class_final_name_ul = [prob_final_class_names_ul[x] for x in class_final_index_ul]

#Writing results in the output

df_out = df_feat_ul_out
print(df_out.shape)
print(len(class_final_name_ul))
print(len(prob_final_ul))


df_out['predicted_class'] = class_final_name_ul
df_out['predicted_class_proba'] = class_final_proba_ul
#test_data_withclass = df_out

#'''
probs_header = prob_final_class_names_ul + '_prob'

prob_pd_ul = pd.DataFrame(prob_final_ul,columns=probs_header,index=df_out.index)

prob_h_pd_ul = pd.DataFrame(test_Y_proba_hierarchical, columns=['prob_Periodic','prob_Stochastic','prob_Transient'],index=df_out.index)

test_data_withclass = df_out.join(prob_pd_ul).join(prob_h_pd_ul)

test_data_withclass.to_csv(class_output)

#'''
