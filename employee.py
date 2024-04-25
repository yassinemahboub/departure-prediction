import streamlit as st
import numpy as np

def predict_employee_leaving(satisfaction_level, average_monthly_hours, promotion_last_5years, salary):
    # Assign the coefficients
    const_coef = 0.10971338
    satisfaction_level_coef = -3.941546
    average_monthly_hours_coef = 0.001952
    promotion_last_5years_coef = 1.239898

    if salary == 'high':
        salary_high_coef = -1.110838
        salary_low_coef = 0.0
        salary_medium_coef = 0.0
    elif salary == 'low':
        salary_high_coef = 0.0
        salary_low_coef = 0.840478
        salary_medium_coef = 0.0
    else:  # 'medium'
        salary_high_coef = 0.0
        salary_low_coef = 0.0
        salary_medium_coef = 0.394705

    # Calculate the linear prediction
    linear_prediction = const_coef \
                        + satisfaction_level_coef * satisfaction_level \
                        + average_monthly_hours_coef * average_monthly_hours \
                        + promotion_last_5years_coef * promotion_last_5years \
                        + salary_high_coef * (salary == 'high') \
                        + salary_low_coef * (salary == 'low') \
                        + salary_medium_coef * (salary == 'medium')

    # Calculate the predicted probabilities using the sigmoid function
    predicted_probability = 1 / (1 + np.exp(-linear_prediction))

    # Apply the binary threshold of 0.35
    y_test_hat = (predicted_probability >= 0.35).astype(int)
    
    return predicted_probability, y_test_hat

def main():
    st.title("Employee Attrition Prediction Model")
    
    
    # Custom CSS to inject into the HTML
    button_css = """
    <style>
    div.stButton > button:first-child {
        background-color: #03AC13;
        color: white;
    }
    </style>
    """
    st.markdown(button_css, unsafe_allow_html=True)
    

    # Input fields
    satisfaction_level = st.slider("What's the satisfaction level (from 1 to 10):", min_value=1, max_value=10, value=5) / 10
    average_monthly_hours = st.slider("What's the average monthly hours:", min_value=80, max_value=320, value=160)
    promotion_last_5years = st.radio("Did he or she got a promotion in the last 5 years?", options=["Yes", "No"], index=1)
    salary = st.selectbox("Enter salary level (high, medium, or low):", options=['high', 'medium', 'low']).lower()

    # Convert promotion_last_5years to 0 or 1
    promotion_last_5years = 1 if promotion_last_5years == "Yes" else 0

    if st.button("Predict Employee Departure"):
        probability, prediction = predict_employee_leaving(satisfaction_level, average_monthly_hours, promotion_last_5years, salary)
        st.subheader("Prediction Results")
        st.write("Probability of leaving: {:.2%}".format(probability))
        st.write("Will the employee leave? : ", "Yes" if prediction else "No")

if __name__ == "__main__":
    main()
