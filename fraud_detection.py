import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score, roc_auc_score
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import cross_val_score
import warnings
warnings.filterwarnings('ignore')
import joblib

def load_and_preprocess_data(sample_size=100000):  
    print("Loading datasets...")
    train_df = pd.read_csv('fraudTrain.csv').sample(n=sample_size, random_state=42)
    test_df = pd.read_csv('fraudTest.csv').sample(n=sample_size//3, random_state=42)
    
    print("\nTraining set shape:", train_df.shape)
    print("Test set shape:", test_df.shape)
    
    categorical_columns = ['merchant', 'category', 'gender', 'state']
    
    for column in categorical_columns:
        train_df[column] = pd.Categorical(train_df[column]).codes
        test_df[column] = pd.Categorical(test_df[column]).codes
    
    train_df['trans_date_trans_time'] = pd.to_datetime(train_df['trans_date_trans_time'])
    test_df['trans_date_trans_time'] = pd.to_datetime(test_df['trans_date_trans_time'])
    
    train_df['hour'] = train_df['trans_date_trans_time'].dt.hour
    test_df['hour'] = test_df['trans_date_trans_time'].dt.hour
    
    columns_to_drop = ['trans_date_trans_time', 'first', 'last', 'street', 'city', 'zip', 'lat', 'long', 'job', 'dob', 'trans_num', 'unix_time']
    train_df = train_df.drop(columns_to_drop, axis=1)
    test_df = test_df.drop(columns_to_drop, axis=1)
    
    X_train = train_df.drop('is_fraud', axis=1)
    y_train = train_df['is_fraud']
    X_test = test_df.drop('is_fraud', axis=1)
    y_test = test_df['is_fraud']
    
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    return X_train_scaled, X_test_scaled, y_train, y_test

def train_and_evaluate_models(X_train, X_test, y_train, y_test):
    models = {
        'Logistic Regression': LogisticRegression(random_state=42, max_iter=500),
        'Decision Tree': DecisionTreeClassifier(random_state=42, max_depth=10),  
        'Random Forest': RandomForestClassifier(random_state=42, n_estimators=50, max_depth=10)  
    }
    
    results = {}
    
    for name, model in models.items():
        print(f"\nTraining {name}...")
        model.fit(X_train, y_train)
        
        y_pred = model.predict(X_test)
        
        accuracy = accuracy_score(y_test, y_pred)
        roc_auc = roc_auc_score(y_test, y_pred)
        
        cv_scores = cross_val_score(model, X_train, y_train, cv=3)  
    
        results[name] = {
            'model': model,
            'accuracy': accuracy,
            'roc_auc': roc_auc,
            'cv_scores': cv_scores,
            'predictions': y_pred
        }
        
        print(f"{name} Results:")
        print(f"Accuracy: {accuracy:.4f}")
        print(f"ROC AUC: {roc_auc:.4f}")
        print(f"Cross-validation scores (mean ± std): {cv_scores.mean():.4f} ± {cv_scores.std():.4f}")
        print("\nClassification Report:")
        print(classification_report(y_test, y_pred))
        
        plt.figure(figsize=(8, 6))
        cm = confusion_matrix(y_test, y_pred)
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
        plt.title(f'Confusion Matrix - {name}')
        plt.ylabel('True Label')
        plt.xlabel('Predicted Label')
        plt.savefig(f'confusion_matrix_{name.lower().replace(" ", "_")}.png')
        plt.close()
    
    return results

def save_best_model(model, scaler):
    print("Saving the model and scaler...")
    joblib.dump(model, 'fraud_detection_model.joblib')
    joblib.dump(scaler, 'scaler.joblib')

def predict_fraud(transaction_data):
    """
    Predict if a credit card transaction is fraudulent.
    
    Parameters:
    transaction_data (dict): Dictionary containing transaction information
        Required keys:
        - 'amount': float (transaction amount)
        - 'merchant': str (merchant name)
        - 'category': str (transaction category)
        - 'gender': str (cardholder gender)
        - 'state': str (transaction state)
        - 'hour': int (hour of transaction, 0-23)
    
    Returns:
    tuple: (prediction (0: legitimate, 1: fraud), probability of fraud)
    """
    try:
        # Load the model and scaler
        model = joblib.load('fraud_detection_model.joblib')
        scaler = joblib.load('scaler.joblib')
        
        # Convert categorical variables
        merchant_map = pd.Categorical(pd.read_csv('fraudTrain.csv')['merchant'].unique()).categories
        category_map = pd.Categorical(pd.read_csv('fraudTrain.csv')['category'].unique()).categories
        gender_map = pd.Categorical(pd.read_csv('fraudTrain.csv')['gender'].unique()).categories
        state_map = pd.Categorical(pd.read_csv('fraudTrain.csv')['state'].unique()).categories
        
        # Create a DataFrame with the transaction data
        df = pd.DataFrame([{
            'amt': float(transaction_data['amount']),
            'merchant': merchant_map.get_loc(transaction_data['merchant']) if transaction_data['merchant'] in merchant_map else -1,
            'category': category_map.get_loc(transaction_data['category']) if transaction_data['category'] in category_map else -1,
            'gender': gender_map.get_loc(transaction_data['gender']) if transaction_data['gender'] in gender_map else -1,
            'state': state_map.get_loc(transaction_data['state']) if transaction_data['state'] in state_map else -1,
            'hour': int(transaction_data['hour'])
        }])
        
        # Scale the features
        scaled_features = scaler.transform(df)
        
        # Make prediction
        prediction = model.predict(scaled_features)[0]
        probability = model.predict_proba(scaled_features)[0][1]
        
        return prediction, probability
        
    except Exception as e:
        print(f"Error in prediction: {str(e)}")
        return None, None

def main():
    # Load and preprocess data with sampling
    X_train, X_test, y_train, y_test = load_and_preprocess_data(sample_size=100000)
    
    # Train and evaluate models
    results = train_and_evaluate_models(X_train, X_test, y_train, y_test)
    
    # Get the best model (Decision Tree)
    best_model = results['Decision Tree']['model']
    
    # Get feature names from training data
    train_df = pd.read_csv('fraudTrain.csv').sample(n=100000, random_state=42)
    categorical_columns = ['merchant', 'category', 'gender', 'state']
    
    for column in categorical_columns:
        train_df[column] = pd.Categorical(train_df[column]).codes
    
    train_df['hour'] = pd.to_datetime(train_df['trans_date_trans_time']).dt.hour
    features_df = train_df[['amt'] + categorical_columns + ['hour']]
    
    # Save the model and scaler
    scaler = StandardScaler()
    scaler.fit(features_df)
    save_best_model(best_model, scaler)
    
    # Example of how to use the prediction function
    example_transaction = {
        'amount': 1000.0,
        'merchant': 'Amazon',
        'category': 'shopping',
        'gender': 'F',
        'state': 'CA',
        'hour': 14
    }
    
    prediction, probability = predict_fraud(example_transaction)
    if prediction is not None:
        print("\nExample Transaction Prediction:")
        print(f"Transaction Amount: ${example_transaction['amount']}")
        print(f"Merchant: {example_transaction['merchant']}")
        print(f"Category: {example_transaction['category']}")
        print(f"Result: {'Fraudulent' if prediction == 1 else 'Legitimate'}")
        print(f"Probability of fraud: {probability:.2%}")
    
    model_names = list(results.keys())
    accuracies = [results[name]['accuracy'] for name in model_names]
    roc_aucs = [results[name]['roc_auc'] for name in model_names]
    
    plt.figure(figsize=(10, 6))
    x = np.arange(len(model_names))
    width = 0.35
    
    plt.bar(x - width/2, accuracies, width, label='Accuracy')
    plt.bar(x + width/2, roc_aucs, width, label='ROC AUC')
    
    plt.xlabel('Models')
    plt.ylabel('Score')
    plt.title('Model Comparison')
    plt.xticks(x, model_names, rotation=45)
    plt.legend()
    plt.tight_layout()
    plt.savefig('model_comparison.png')
    plt.close()

if __name__ == "__main__":
    main()
