# Spaceship Titanic Prediction

## 🚀 Project Overview
This project aims to predict whether a passenger was transported to an alternate dimension during the Titanic's cosmic voyage. Based on the **Kaggle Spaceship Titanic competition**, this notebook implements a complete machine learning pipeline including data preprocessing, feature engineering, model training, hyperparameter tuning, and submission generation.

## 📊 Dataset
The dataset contains personal records for approximately 9,000 passengers, including:
- **PassengerId**: Unique identifier
- **HomePlanet**: Planet of origin
- **CryoSleep**: Whether the passenger entered suspended animation
- **Cabin**: Cabin number (Deck/Num/Side)
- **Destination**: Destination planet
- **Age**: Passenger age
- **VIP**: Whether the passenger paid for special VIP services
- **Spending Columns**: RoomService, FoodCourt, ShoppingMall, Spa, VRDeck
- **Name**: Passenger name
- **Transported**: Target variable (Boolean)

## 🛠️ Methodology

### 1. Data Preprocessing
- **Missing Value Imputation**:
  - `HomePlanet`: Filled using Group logic and Deck correlations (e.g., Decks A, B, C, T → Europa; Deck G → Earth).
  - `CryoSleep`: Inferred from spending habits (Spending > 0 → False; Spending == 0 → True).
  - `VIP`: Inferred from Deck and HomePlanet (Deck G & Earth → False).
  - `Spending Columns`: Filled with 0.0 if CryoSleep=True, otherwise median.
  - `Age`: Filled with median.
  - `Cabin`: Filled with placeholder 'U/U/U'.
- **Feature Engineering**:
  - **Group & Group_Size**: Extracted from `PassengerId` to determine travel party size.
  - **Is_Solo**: Binary feature indicating if the passenger is traveling alone.
  - **Total_Spending**: Sum of all luxury spending columns.
  - **No_Spending**: Binary feature indicating zero total spending.
  - **Deck & Side**: Extracted from the `Cabin` string.

### 2. Encoding & Scaling
- **One-Hot Encoding**: Applied to categorical variables (`HomePlanet`, `Destination`, `Deck`, `Side`).
- **Standard Scaling**: Applied to numerical features (`Age`, spending columns, `Total_Spending`, `Group_Size`).
- **Boolean Conversion**: True/False columns converted to 1/0.

### 3. Modeling
- **Algorithm**: Logistic Regression.
- **Hyperparameter Tuning**: Used `GridSearchCV` to optimize `C` (regularization) and `solver`.
  - **Best Parameters**: `C=10`, `solver='lbfgs'`.
  - **Best Cross-Validation Score**: ~79.32%.
- **Final Test Accuracy**: ~78.66%.

## 📈 Results
| Metric | Score |
| :--- | :--- |
| **Baseline Accuracy** | 78.75% |
| **Tuned CV Score** | 79.32% |
| **Final Test Accuracy** | 78.66% |

## 📁 File Structure
```
├── spaceship-titanic/             # Data folder 
│   ├── train.csv                  # Training data
│   └── test.csv                   # Test data for submission
├── LICENSE                        # Project license
├── README.md                      # This file - Project documentation
├── Spaceship Titanic.ipynb        # Main Jupyter Notebook with complete analysis
├── Spaceship_titanic_model.pkl    # Saved trained Logistic Regression model
├── bash.bat                       # Bash/Batch script for automation
├── requirements.txt               # Python dependencies
├── spaceship_titanic_documentation.docx  # Detailed project documentation
└── sub.csv                        # Final submission file for Kaggle
```

## 🚀 Usage

### Prerequisites
Ensure you have the necessary libraries installed:
```bash
pip install -r requirements.txt
```

### Running the Notebook
1. Open `Spaceship Titanic.ipynb` in Jupyter Notebook or Jupyter Lab.
2. Run all cells sequentially.
3. The notebook will automatically generate:
   - `sub.csv`: Ready for Kaggle submission.
   - `Spaceship_titanic_model.pkl`: Serialized model for future use.

### Key Code Snippets
**Feature Engineering (Group Size):**
```python
df['Group'] = df['PassengerId'].str.split('_').str[0]
df['Group_Size'] = df['Group'].map(df['Group'].value_counts())
df['Is_Solo'] = df['Group_Size'] == 1
```

**Model Training & Tuning:**
```python
grid_search = GridSearchCV(estimator=base_lr, param_grid=param_grid, cv=5, scoring='accuracy')
grid_search.fit(x_train, y_train)
best_model = grid_search.best_estimator_
```

### Using the Saved Model
```python
import pickle

# Load the trained model
with open('Spaceship_titanic_model.pkl', 'rb') as file:
    model = pickle.load(file)

# Make predictions
predictions = model.predict(new_data)
```

## 📝 Notes
- **Column Alignment**: A crucial step in the preprocessing of the test set involves reindexing columns to match the training set exactly (`test_df.reindex(columns=x_train.columns, fill_value=0)`), ensuring compatibility during prediction.
- **Model Persistence**: The final model is saved using `pickle` for deployment or future inference without retraining.
- **Data Location**: Training and test data are stored in the `spaceship-titanic/` folder.

## 📄 License
This project is for educational purposes & protected under MIT license.

## 🤝 Contributing
Feel free to fork this repository and submit pull requests for any improvements or suggestions.

## 📧 Contact
For questions or suggestions, please open an issue in this repository.
