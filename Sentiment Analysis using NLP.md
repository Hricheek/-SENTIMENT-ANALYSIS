```python
!pip install scikit-learn pandas
# Import necessary libraries
import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import accuracy_score, classification_report
```

    Requirement already satisfied: scikit-learn in c:\users\user\anaconda3\lib\site-packages (1.2.2)
    Requirement already satisfied: pandas in c:\users\user\anaconda3\lib\site-packages (2.1.4)
    Requirement already satisfied: numpy>=1.17.3 in c:\users\user\anaconda3\lib\site-packages (from scikit-learn) (1.26.4)
    Requirement already satisfied: scipy>=1.3.2 in c:\users\user\anaconda3\lib\site-packages (from scikit-learn) (1.11.4)
    Requirement already satisfied: joblib>=1.1.1 in c:\users\user\anaconda3\lib\site-packages (from scikit-learn) (1.2.0)
    Requirement already satisfied: threadpoolctl>=2.0.0 in c:\users\user\anaconda3\lib\site-packages (from scikit-learn) (2.2.0)
    Requirement already satisfied: python-dateutil>=2.8.2 in c:\users\user\anaconda3\lib\site-packages (from pandas) (2.8.2)
    Requirement already satisfied: pytz>=2020.1 in c:\users\user\anaconda3\lib\site-packages (from pandas) (2023.3.post1)
    Requirement already satisfied: tzdata>=2022.1 in c:\users\user\anaconda3\lib\site-packages (from pandas) (2023.3)
    Requirement already satisfied: six>=1.5 in c:\users\user\anaconda3\lib\site-packages (from python-dateutil>=2.8.2->pandas) (1.16.0)
    


```python
# Sample data
data = {'Review': ['I love this product','Worst experience ever','Decent quality','Absolutely fantastic','Not worth the price','Could be better', 'Best purchase I made'],
    'Sentiment': ['Positive', 'Negative', 'Neutral', 'Positive', 'Negative','Neutral', 'Positive']}

# Load data into a DataFrame
df = pd.DataFrame(data)
df.head()
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>Review</th>
      <th>Sentiment</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>I love this product</td>
      <td>Positive</td>
    </tr>
    <tr>
      <th>1</th>
      <td>Worst experience ever</td>
      <td>Negative</td>
    </tr>
    <tr>
      <th>2</th>
      <td>Decent quality</td>
      <td>Neutral</td>
    </tr>
    <tr>
      <th>3</th>
      <td>Absolutely fantastic</td>
      <td>Positive</td>
    </tr>
    <tr>
      <th>4</th>
      <td>Not worth the price</td>
      <td>Negative</td>
    </tr>
  </tbody>
</table>
</div>




```python

# Vectorize text data
vectorizer = CountVectorizer()
X = vectorizer.fit_transform(df['Review'])  # Features
y = df['Sentiment'].map({'Positive': 1, 'Negative': 0, 'Neutral': 2})  # Encode target labels

# Split data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
df.head()
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>Review</th>
      <th>Sentiment</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>I love this product</td>
      <td>Positive</td>
    </tr>
    <tr>
      <th>1</th>
      <td>Worst experience ever</td>
      <td>Negative</td>
    </tr>
    <tr>
      <th>2</th>
      <td>Decent quality</td>
      <td>Neutral</td>
    </tr>
    <tr>
      <th>3</th>
      <td>Absolutely fantastic</td>
      <td>Positive</td>
    </tr>
    <tr>
      <th>4</th>
      <td>Not worth the price</td>
      <td>Negative</td>
    </tr>
  </tbody>
</table>
</div>




```python
# Train the Naive Bayes model
model = MultinomialNB()
model.fit(X_train, y_train)

# Predict sentiments for the test data
y_pred = model.predict(X_test)

```


```python
# Make predictions
y_pred = model.predict(X_test)

# Evaluate the model
print("Accuracy:", accuracy_score(y_test, y_pred))
print("\nClassification Report:\n", classification_report(y_test, y_pred))
```

    Accuracy: 0.5
    
    Classification Report:
                   precision    recall  f1-score   support
    
               0       0.00      0.00      0.00         1
               1       0.50      1.00      0.67         1
    
        accuracy                           0.50         2
       macro avg       0.25      0.50      0.33         2
    weighted avg       0.25      0.50      0.33         2
    
    

    C:\Users\user\anaconda3\Lib\site-packages\sklearn\metrics\_classification.py:1344: UndefinedMetricWarning: Precision and F-score are ill-defined and being set to 0.0 in labels with no predicted samples. Use `zero_division` parameter to control this behavior.
      _warn_prf(average, modifier, msg_start, len(result))
    C:\Users\user\anaconda3\Lib\site-packages\sklearn\metrics\_classification.py:1344: UndefinedMetricWarning: Precision and F-score are ill-defined and being set to 0.0 in labels with no predicted samples. Use `zero_division` parameter to control this behavior.
      _warn_prf(average, modifier, msg_start, len(result))
    C:\Users\user\anaconda3\Lib\site-packages\sklearn\metrics\_classification.py:1344: UndefinedMetricWarning: Precision and F-score are ill-defined and being set to 0.0 in labels with no predicted samples. Use `zero_division` parameter to control this behavior.
      _warn_prf(average, modifier, msg_start, len(result))
    


```python


```


```python

```


```python


```


```python

```
