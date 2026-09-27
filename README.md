💧 Water Quality Risk Classification
An end-to-end, zero-cost machine learning prototype developed as a 10-day capstone project. This system evaluates physicochemical water parameters and classifies samples into safe or high-risk categories to assist in environmental safety monitoring.

🚀 Project Overview
Access to clean drinking water is vital. This project provides an automated, lightweight machine learning solution that takes core water quality indicators as inputs, processes them through an imputation and scaling pipeline, evaluates them using a foundational Decision Tree classifier, and exposes predictions through an interactive web interface.

🛠️ Tech Stack & Tools
Programming Language: Python 3.8+

Machine Learning: Scikit-Learn (Decision Tree Classifier, StandardScaler, SimpleImputer)

Data Manipulation: Pandas, NumPy

Model Persistence: Joblib

Web UI Framework: Streamlit

Development Environment: Visual Studio Code

📊 Features & ArchitectureData Preprocessing Pipeline:
Automatically handles missing values via median imputation and normalizes features using standard scaling.Foundational Model: Utilizes a supervised Decision Tree classifier optimized for interpretability and zero-cost execution.   Interactive Web Application: Built using Streamlit, allowing users to modify water parameters in real time and view instant risk predictions.

📂 Repository Structure
water-quality-capstone/
├── model_training.py       # Trains model, handles preprocessing, and saves artifacts
├── app.py                  # Streamlit web application interface
├── requirements.txt        # Project dependencies
├── water_model.pkl         # Serialized decision tree model (generated after training)
├── scaler.pkl              # Serialized standard scaler (generated after training)
├── imputer.pkl             # Serialized missing-value imputer (generated after training)
└── README.md               # Project documentation

⚙️ Installation & Local Setup
Follow these steps to set up and run the project locally on your machine:

Clone the repository:
git clone https://github.com/your-username/water-quality-capstone.git
cd water-quality-capstone

Install dependencies:
pip install -r requirements.txt

Train the model and generate artifacts:
Run the training script to build the model and save the .pkl files:
python model_training.py

Launch the Streamlit web application:
streamlit run app.py
(Streamlit will automatically open a local web page in your browser, typically at http://localhost:8501)

📈 Model Performance & EvaluationMetrics Tracked:
Accuracy, Precision, Recall, F1-score, and Confusion Matrix.   Interpretation: The model minimizes false negatives on high-risk water samples, ensuring critical pollution alerts are not missed.


📄 License & Acknowledgments
Developed as part of the Learn Depth Academy Capstone Program (Environment Domain - Problem 09). Free and open-source tools were used exclusively.
