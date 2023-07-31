'''
This module provide several functions
to build a customer churn prediction system.

Author: Mustafa Alturki
Date: 19/07/2023
'''

# import libraries
import os
from sklearn.metrics import plot_roc_curve, classification_report
from sklearn.model_selection import GridSearchCV
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
import pandas as pd
import joblib
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
sns.set()


os.environ['QT_QPA_PLATFORM'] = 'offscreen'


def import_data(pth):
    '''
    returns data_frame for the csv found at pth

    input:
            pth: a path to the csv
    output:
            data_frame: pandas data_frame
    '''
    data_frame = pd.read_csv(pth)
    return data_frame


def preprocess(data_frame):
    '''
    create Churn column in df based on Attrition_Flag column
    input:
            data_frame: pandas data_frame

    output:
            data_frame: pandas data_frame with new column Churn with 0 or 1 values
    '''
    data_frame['Churn'] = data_frame['Attrition_Flag'].apply(
        lambda val: 0 if val == "Existing Customer" else 1)
    return data_frame


def perform_eda(data_frame):
    '''
    perform eda on df and save figures to images folder
    input:
            data_frame: pandas data_frame

    output:
            None
    '''
    figsize = (20, 10)
    col_lst = [
        'Churn',
        'Customer_Age',
        'Marital_Status',
        'Total_Trans_Ct',
        'cols_heatmap']
    for col in col_lst:
        plt.figure(figsize=figsize)
        if col in ['Churn', 'Customer_Age']:
            data_frame[col].hist()
        elif col == 'Marital_Status':
            data_frame[col].value_counts('normalize').plot(kind='bar')
        elif col == 'Total_Trans_Ct':
            # Show distributions of 'Total_Trans_Ct' and add a smooth curve
            # obtained using a kernel density estimate
            sns.histplot(data_frame['Total_Trans_Ct'], stat='density', kde=True)
        elif col == 'cols_heatmap':
            sns.heatmap(data_frame.corr(), annot=False, cmap='Dark2_r', linewidths=2)
        plt.savefig(f'images/eda/{col}.png')
        plt.close()


def encoder_helper(data_frame, category_lst, response):
    '''
    helper function to turn each categorical column into a new column with
    propotion of churn for each category - associated with cell 15 from the notebook

    input:
            data_frame: pandas data_frame
            category_lst: list of columns that contain categorical features
            response: string of response name [optional argument that could be
            used for naming variables or index y column]

    output:
            data_frame: pandas data_frame with new columns for each categorical feature
    '''

    # gender encoded column
    for feature in category_lst:
        feature_lst = []
        feature_groups = data_frame.groupby(feature).mean()[response]

        for val in data_frame[feature]:
            feature_lst.append(feature_groups.loc[val])

        data_frame[feature + '_' + response] = feature_lst
    return data_frame


def perform_feature_engineering(data_frame, response):
    '''
    input:
              data_frame: pandas data_frame
              response: string of response name [optional argument that could be
              used for naming variables or index y column]

    output:
              x_train: X training data
              x_test: X testing data
              y_train: y training data
              y_test: y testing data
    '''
    cat_columns = [
        'Gender',
        'Education_Level',
        'Marital_Status',
        'Income_Category',
        'Card_Category'
    ]
    keep_cols = [
        'Customer_Age',
        'Dependent_count',
        'Months_on_book',
        'Total_Relationship_Count',
        'Months_Inactive_12_mon',
        'Contacts_Count_12_mon',
        'Credit_Limit',
        'Total_Revolving_Bal',
        'Avg_Open_To_Buy',
        'Total_Amt_Chng_Q4_Q1',
        'Total_Trans_Amt',
        'Total_Trans_Ct',
        'Total_Ct_Chng_Q4_Q1',
        'Avg_Utilization_Ratio',
        'Gender_Churn',
        'Education_Level_Churn',
        'Marital_Status_Churn',
        'Income_Category_Churn',
        'Card_Category_Churn']
    data_frame = encoder_helper(data_frame, cat_columns, response)

    x_s = pd.data_frame()
    y_s = data_frame[response]
    x_s[keep_cols] = data_frame[keep_cols]
    # train test split
    x_train, x_test, y_train, y_test = train_test_split(
        x_s, y_s, test_size=0.3, random_state=42)
    return x_train, x_test, y_train, y_test


def classification_report_image(y_train,
                                y_test,
                                y_train_preds_lr,
                                y_train_preds_rf,
                                y_test_preds_lr,
                                y_test_preds_rf):
    '''
    produces classification report for training and testing results and stores report as image
    in images folder
    input:
            y_train: training response values
            y_test:  test response values
            y_train_preds_lr: training predictions from logistic regression
            y_train_preds_rf: training predictions from random forest
            y_test_preds_lr: test predictions from logistic regression
            y_test_preds_rf: test predictions from random forest

    output:
             None
    '''

    classifiers_predictions = {
        'Random Forest': {
            'train_pred': y_train_preds_rf,
            'test_pred': y_test_preds_rf
        },
        'Logistic Regression': {
            'train_pred': y_train_preds_lr,
            'test_pred': y_test_preds_lr
        }
    }
    for classifier, predictions in classifiers_predictions.items():
        plt.rc('figure', figsize=(5, 5))
        for split in ['train', 'test']:
            ground_truth = y_train if split == 'train' else y_test
            plt.text(0.01, 1.25, str(classifier + ' ' + split), {
                'fontsize': 10}, fontproperties='monospace')
            plt.text(0.01,
                     0.05,
                     str(classification_report(ground_truth,
                                               predictions[split + '_pred'])),
                     {'fontsize': 10},
                     fontproperties='monospace')
        plt.axis('off')
        plt.savefig(
            f'images/results/{classifier.replace(" ", "_")}_results.png')


def feature_importance_plot(model, x_data, output_pth):
    '''
    creates and stores the feature importances in pth
    input:
            model: model object containing feature_importances_
            X_data: pandas data_frame of X values
            output_pth: path to store the figure

    output:
             None
    '''

    # Calculate feature importances
    importances = model.best_estimator_.feature_importances_
    # Sort feature importances in descending order
    indices = np.argsort(importances)[::-1]

    # Rearrange feature names so they match the sorted feature importances
    names = [x_data.columns[i] for i in indices]

    # Create plot
    plt.figure(figsize=(20, 5))

    # Create plot title
    plt.title("Feature Importance")
    plt.ylabel('Importance')

    # Add bars
    plt.bar(range(x_data.shape[1]), importances[indices])

    # Add feature names as x-axis labels
    plt.xticks(range(x_data.shape[1]), names, rotation=90)
    plt.savefig(f'{output_pth}/feature_importances.png')
    plt.close()


def train_models(x_train, x_test, y_train, y_test):
    '''
    train, store model results: images + scores, and store models
    input:
              x_train: X training data
              X_test: X testing data
              y_train: y training data
              y_test: y testing data
    output:
              None
    '''
    # grid search
    rfc = RandomForestClassifier(random_state=42)
    # Use a different solver if the default 'lbfgs' fails to converge
    # Reference:
    # https://scikit-learn.org/stable/modules/linear_model.html#logistic-regression
    lrc = LogisticRegression(solver='lbfgs', max_iter=3000)

    param_grid = {
        'n_estimators': [200, 500],
        'max_features': ['auto', 'sqrt'],
        'max_depth': [4, 5, 100],
        'criterion': ['gini', 'entropy']
    }

    cv_rfc = GridSearchCV(estimator=rfc, param_grid=param_grid, cv=5)
    cv_rfc.fit(x_train, y_train)

    lrc.fit(x_train, y_train)

    y_train_preds_rf = cv_rfc.best_estimator_.predict(x_train)
    y_test_preds_rf = cv_rfc.best_estimator_.predict(x_test)

    y_train_preds_lr = lrc.predict(x_train)
    y_test_preds_lr = lrc.predict(x_test)

    # plot roc curves
    lrc_plot = plot_roc_curve(lrc, x_test, y_test)
    plt.figure(figsize=(15, 8))
    axis = plt.gca()
    _ = plot_roc_curve(cv_rfc.best_estimator_,
                              x_test, y_test, ax=axis, alpha=0.8)
    lrc_plot.plot(ax=axis, alpha=0.8)
    plt.show()
    plt.savefig('images/results/roc_curves_results.png')

    classification_report_image(y_train,
                                y_test,
                                y_train_preds_lr,
                                y_train_preds_rf,
                                y_test_preds_lr,
                                y_test_preds_rf)
    feature_importance_plot(cv_rfc, x_train, 'images/results')
    # save best model
    joblib.dump(cv_rfc.best_estimator_, './models/rfc_model.pkl')
    joblib.dump(lrc, './models/logistic_model.pkl')


if __name__ == "__main__":
    raw_data_frame = import_data(r"./data/bank_data.csv")
    processedDataFrame = preprocess(raw_data_frame)
    perform_eda(processedDataFrame)
    x_train, x_test, y_train, y_test = perform_feature_engineering(processedDataFrame, "Churn")
    train_models(x_train, x_test, y_train, y_test)
