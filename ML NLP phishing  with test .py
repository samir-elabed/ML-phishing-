import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from wordcloud import WordCloud

# Function to generate and display a word cloud
def generate_word_cloud(text):
    wordcloud = WordCloud(width=800, height=400, background_color='white').generate(text)
    plt.figure(figsize=(10, 5))
    plt.imshow(wordcloud, interpolation='bilinear')
    plt.axis('off')
    plt.show()

# Load your dataset (replace 'your_dataset.csv' with your actual dataset)
data = pd.read_csv('GMAIl.csv', low_memory=False) 

# Replace NaN values with empty strings in the 'text' column before vectorization
data['text'] = data['text'].fillna('')

# Remove rows where the 'label' is missing
data.dropna(subset=['label'], inplace=True)

# Assume your dataset has 'text' and 'label' columns
X = data['text'].values
y = data['label'].values

# Split the dataset into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Convert text data to numerical features using CountVectorizer
vectorizer = CountVectorizer(max_features=1000)  # You can adjust max_features based on your needs
X_train_vectorized = vectorizer.fit_transform(X_train)
X_test_vectorized = vectorizer.transform(X_test)

# Build a classifier (e.g., Multinomial Naive Bayes)
classifier = MultinomialNB()
classifier.fit(X_train_vectorized, y_train)

# Make predictions on the test set
y_pred = classifier.predict(X_test_vectorized)

# Evaluate the classifier
accuracy = accuracy_score(y_test, y_pred)
print(f'Test Accuracy: {accuracy:.4f}')

# Display additional evaluation metrics
print("Classification Report:")
print(classification_report(y_test, y_pred))

# Display confusion matrix
conf_matrix = confusion_matrix(y_test, y_pred)
print("Confusion Matrix:")
print(conf_matrix)

# (Optional) Plot confusion matrix
plt.figure(figsize=(8, 6))
plt.imshow(conf_matrix, interpolation='nearest', cmap=plt.cm.Blues)
plt.title('Confusion Matrix')
plt.colorbar()
plt.xticks([0, 1], ['Legitimate', 'Phishing'])
plt.yticks([0, 1], ['Legitimate', 'Phishing'])
plt.ylabel('True label')
plt.xlabel('Predicted label')
plt.show()


# Generate word clouds for Legitimate and Phishing emails
legitimate_emails_text = " ".join(data.loc[data['label'] == 'Legitimate', 'text'])
phishing_emails_text = " ".join(data.loc[data['label'] == 'Phishing', 'text'])

print("Word Cloud for Legitimate Emails:")
generate_word_cloud(legitimate_emails_text)

print("Word Cloud for Phishing Emails:")
generate_word_cloud(phishing_emails_text)

# Example: Test new emails

new_emails = ["please find attached the invoice below and download and run it to check that it is working for you ","Your account has been breached. Verify your details at http://phishy-link.com immediately","Hello Samir elabed, These are the weekly statistics for your ad: Passionate Programming Tutor with Extensive Expertise: De.... You have not received any visits this week..Your ad is not positioned well. You have not received any messages.","Hello Samir,Your latest bill is ready to view. To see it, please go to bt.com/yourlatestbill or click View your bill.You don't need to do anything. We'll take your Direct Debit as normal on the date shown on your bill.Thanks, BT Customer SupportAny questions? We're here to help.Account number: ******7637","Dear Valued Customer, We have detected unusual activity on your account that suggests an unauthorized access attempt. As part of our ongoing commitment to protect your personal and financial information, we require you to verify your account immediately. Please visit the following link to verify your information as soon as possible: https://your-bank.verify-login.com Failure to complete the verification within 24 hours will result in temporary account suspension as a security precaution. Thank you for your prompt attention to this matter. Sincerely, Customer Support Team"]
new_emails_vectorized = vectorizer.transform(new_emails)
new_predictions = classifier.predict(new_emails_vectorized)

# Display the results for new emails
for email, prediction in zip(new_emails, new_predictions):
    print("\n")
    print(f"Email: '{email}'\n \n Predicted Label: '{prediction}'\n")