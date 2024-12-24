import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split

def check_transaction(amount, merchant, category, gender, state, hour):
    """
    Check if a credit card transaction is potentially fraudulent.
    """
    
    # Load training data
    print("Loading and preparing data...")
    train_df = pd.read_csv('fraudTrain.csv')
    
    # Add risk factors
    def is_suspicious_hour(h):
        return 1 if (h >= 23) or (h <= 4) else 0
    
    def is_suspicious_amount(amt, mean_amt, std_amt):
        return 1 if amt > (mean_amt + 2 * std_amt) else 0
    
    # Prepare the input transaction
    transaction = pd.DataFrame({
        'amt': [amount],
        'merchant': [merchant],
        'category': [category],
        'gender': [gender],
        'state': [state],
        'hour': [hour]
    })
    
    # Add hour-based risk factor
    train_df['hour'] = pd.to_datetime(train_df['trans_date_trans_time']).dt.hour
    train_df['suspicious_hour'] = train_df['hour'].apply(is_suspicious_hour)
    transaction['suspicious_hour'] = transaction['hour'].apply(is_suspicious_hour)
    
    # Add amount-based risk factor
    mean_amount = train_df['amt'].mean()
    std_amount = train_df['amt'].std()
    train_df['suspicious_amount'] = train_df['amt'].apply(lambda x: is_suspicious_amount(x, mean_amount, std_amount))
    transaction['suspicious_amount'] = transaction['amt'].apply(lambda x: is_suspicious_amount(x, mean_amount, std_amount))
    
    # Convert categorical variables
    categorical_columns = ['merchant', 'category', 'gender', 'state']
    encoders = {}
    
    for col in categorical_columns:
        encoders[col] = LabelEncoder()
        train_df[col] = encoders[col].fit_transform(train_df[col])
        try:
            transaction[col] = encoders[col].transform(transaction[col])
        except ValueError:
            # If category not seen in training, mark as suspicious (-1)
            transaction[col] = -1
    
    # Select features for training
    features = ['amt', 'suspicious_hour', 'suspicious_amount'] + categorical_columns + ['hour']
    X = train_df[features]
    y = train_df['is_fraud']
    
    # Split data
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    # Scale the features
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    transaction_scaled = scaler.transform(transaction[features])
    
    # Train a more sophisticated model
    print("Training the model...")
    model = RandomForestClassifier(
        n_estimators=200,           # Increased from 100
        max_depth=20,               # Increased from 15
        min_samples_split=5,        # Decreased from 10
        random_state=42,
        class_weight={0: 1, 1: 10}  # Give more weight to fraud cases
    )
    model.fit(X_train_scaled, y_train)
    
    # Make prediction with threshold adjustment
    raw_prediction = model.predict_proba(transaction_scaled)[0][1]
    
    # Adjust probability based on risk factors
    risk_multiplier = 1.0
    if transaction['suspicious_hour'].iloc[0] == 1:
        risk_multiplier *= 1.2
    if transaction['suspicious_amount'].iloc[0] == 1:
        risk_multiplier *= 1.5
    if merchant == "Unknown":
        risk_multiplier *= 1.3
    
    # Apply risk multiplier
    fraud_probability = min(raw_prediction * risk_multiplier, 1.0)
    prediction = 1 if fraud_probability > 0.5 else 0
    
    # Calculate risk score (0-100)
    risk_score = int(fraud_probability * 100)
    
    # Determine risk level
    if risk_score < 30:
        risk_level = "LOW"
    elif risk_score < 70:
        risk_level = "MEDIUM"
    else:
        risk_level = "HIGH"
    
    # Get feature importance for this prediction
    feature_importance = dict(zip(features, model.feature_importances_))
    top_factors = sorted(feature_importance.items(), key=lambda x: x[1], reverse=True)[:3]
    
    # Prepare risk factors message
    risk_factors = []
    if transaction['suspicious_hour'].iloc[0] == 1:
        risk_factors.append("Unusual transaction hour")
    if transaction['suspicious_amount'].iloc[0] == 1:
        risk_factors.append(f"Amount exceeds normal range (mean: ${mean_amount:.2f})")
    if merchant == "Unknown":
        risk_factors.append("Unknown merchant")
    
    message = f"""
Transaction Assessment:
----------------------
Amount: ${amount:.2f}
Merchant: {merchant}
Category: {category}
Time: {hour:02d}:00
State: {state}
----------------------
Result: {"FRAUDULENT" if prediction == 1 else "LEGITIMATE"}
Risk Level: {risk_level}
Risk Score: {risk_score}/100
Fraud Probability: {fraud_probability:.1%}

Risk Factors:
{chr(10).join(['- ' + factor for factor in risk_factors]) if risk_factors else "No significant risk factors identified"}

Top Contributing Features:
{chr(10).join(['- ' + str(feature) for feature, importance in top_factors])}
"""
    return message

if __name__ == "__main__":
    # Test with a highly suspicious transaction
    print(check_transaction(
        amount=10000.0,         # Very large amount
        merchant="Unknown",     # Unknown merchant
        category="other",       # Vague category
        gender="M",
        state="FL",            # Different state
        hour=2                 # Very late night transaction
    ))
