# -SENTIMENT-ANALYSIS

# Company:CODTECH IT SOLUTIONS

# Name:Hricheek Bhattacharjee

# Intern ID:CT04WB108

# Domain:Data Analysis

# Mentor:Neela Santosh

# Description:
The code implements a sentiment analysis model using a Naïve Bayes classifier to classify customer reviews into Positive, Negative, or Neutral categories. It begins by importing necessary Python libraries such as pandas for handling structured data, CountVectorizer from scikit-learn for text processing, and MultinomialNB for classification. The dataset consists of seven sample reviews with corresponding sentiment labels. These labels are later mapped to numerical values: 1 for Positive, 0 for Negative, and 2 for Neutral.To convert text data into a machine-readable format, the script employs CountVectorizer, which transforms the raw text into a numerical feature matrix based on word frequency. Once vectorized, the dataset is split into training and test subsets, with 80% of the data used for training and 20% for testing. The split ensures that the model is trained on a subset while being evaluated on unseen data.
A Naïve Bayes classifier, specifically MultinomialNB, is chosen for training due to its efficiency in handling text classification tasks. The model is fitted on the training data, meaning it learns patterns between word frequencies and sentiment categories. After training, the model is tested by making predictions on the test dataset.The model's predictions are evaluated using accuracy and a classification report. Accuracy represents the proportion of correctly classified sentiments, while the classification report provides precision, recall, and F1-score metrics for each category. The output shows an accuracy of 50%, meaning the model correctly classified only half of the test samples. Additionally, the classification report indicates that the model failed to predict at least one sentiment category, leading to warnings regarding undefined precision and F1-scores.The low accuracy and warnings stem from the dataset’s small size and class imbalance. With only seven reviews, the dataset does not provide enough diversity for the model to generalize well. Additionally, the Naïve Bayes classifier assumes word independence and may not perform optimally with limited text data. These limitations suggest that increasing the dataset size and balancing sentiment distribution would improve performance.
To address these issues, expanding the dataset with real-world reviews, applying text preprocessing techniques (removing stop words, stemming, and lemmatization), and tuning model hyperparameters could enhance the classifier’s accuracy. Additionally, incorporating a more diverse set of words and improving training-test splits may further optimize performance.Ultimately, while the implementation demonstrates the fundamental workflow of sentiment classification, it highlights the importance of dataset size, class balance, and model selection. A larger dataset, improved feature engineering, and alternative classification models could significantly boost accuracy and mitigate issues observed in the current implementation. 










