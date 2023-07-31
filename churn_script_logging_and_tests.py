'''
This module provide functions to test churn_library functions.

Author: Mustafa Alturki
Date: 19/07/2023
'''
import os
import logging
# import churn_library_solution as cls
import churn_library as cl

logging.basicConfig(
    filename='./logs/churn_library.log',
    level=logging.INFO,
    filemode='w',
    format='%(name)s - %(levelname)s - %(message)s')


def test_import(import_data):
    '''
    test data import - this example is completed for you to assist
	with the other test functions
    '''
    try:
        data_frame = import_data("./data/bank_data.csv")
        logging.info("Testing import_data: SUCCESS")
    except FileNotFoundError as err:
        logging.error("Testing import_eda: The file wasn't found")
        raise err

    try:
        assert data_frame.shape[0] > 0
        assert data_frame.shape[1] > 0
    except AssertionError as err:
        logging.error(
            "Testing import_data: The file doesn't appear to have rows and columns")
        raise err


def test_preprocess(preprocess):
    '''
    test preprocess function
    '''
    data_frame = cl.import_data("./data/bank_data.csv")
    data_frame = preprocess(data_frame)
    if 'Churn' not in data_frame.columns:
        logging.error("Testing preprocess: The column Churn wasn't created")
    else:
        logging.info("Testing preprocess: SUCCESS")


def test_eda(perform_eda):
    '''
    test perform eda function
    '''
    data_frame = cl.import_data("./data/bank_data.csv")
    data_frame = cl.preprocess(data_frame)
    perform_eda(data_frame)
    success = True
    saved_img_list = os.listdir('./images/eda')
    for col in [
        'Churn',
        'Customer_Age',
        'Marital_Status',
        'Total_Trans_Ct',
            'cols_heatmap']:
        if col + '.png' not in saved_img_list:
            logging.error(f"Testing perform_eda: {col}.png wasn't found")
            success = False
    if success:
        logging.info("Testing perform_eda: SUCCESS")


def test_encoder_helper(encoder_helper):
    '''
    test encoder helper
    '''
    cat_columns = [
        'Gender',
        'Education_Level',
        'Marital_Status',
        'Income_Category',
        'Card_Category'
    ]
    response = 'Churn'
    success = True
    data_frame = cl.import_data("./data/bank_data.csv")
    data_frame = cl.preprocess(data_frame)
    data_frame = encoder_helper(data_frame, cat_columns, response)
    for col in cat_columns:
        if col + '_' + response not in data_frame.columns:
            logging.error(
                f"Testing encoder_helper: column {col}_{response} wasn't created")
            success = False
    if success:
        logging.info("Testing encoder_helper: SUCCESS")


def test_perform_feature_engineering(perform_feature_engineering):
    '''
    test perform_feature_engineering
    '''
    data_frame = cl.import_data("./data/bank_data.csv")
    data_frame = cl.preprocess(data_frame)
    x_train, x_test, y_train, y_test = perform_feature_engineering(
        data_frame, "Churn")
    try:
        for arr in x_train, x_test, y_train, y_test:
            for dim in arr.shape:
                assert dim > 0
        logging.info("Testing perform_feature_engineering: SUCCESS")

    except AssertionError:
        logging.error(
            "Testing perform_feature_engineering: Some split doesn't appear to \
				 have rows or columns")


def test_train_models(train_models):
    '''
    test train_models
    '''
    data_frame = cl.import_data("./data/bank_data.csv")
    data_frame = cl.preprocess(data_frame)
    x_train, x_test, y_train, y_test = cl.perform_feature_engineering(
        data_frame, "Churn")
    train_models(x_train, x_test, y_train, y_test)
    success = True
    img_list = os.listdir('./images/results')
    for title in [
        'feature_importances',
        'roc_curves_results',
        'Random_Forest_results',
            'Logistic_Regression_results']:
        if title + '.png' not in img_list:
            success = False
            logging.error(f"Testing train_models: {title}.png wasn't found")
    model_list = os.listdir('./models')
    for model in ['logistic_model', 'rfc_model']:
        if model + '.pkl' not in model_list:
            success = False
            logging.error(f"Testing train_models: {model}.pkl wasn't found")

    if success:
        logging.info("Testing train_models: SUCCESS")


if __name__ == "__main__":
    test_import(cl.import_data)
    test_preprocess(cl.preprocess)
    test_eda(cl.perform_eda)
    test_encoder_helper(cl.encoder_helper)
    test_perform_feature_engineering(cl.perform_feature_engineering)
    test_train_models(cl.train_models)
