import fnmatch
import os
import pandas as pd
import matplotlib.pyplot as plt
import string
from nltk.corpus import stopwords
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import classification_report,confusion_matrix

# f = open('./all_emails.txt', 'w')
# file_number = 0
# # Walk the directories recursively
# for root, dirnames, filenames in os.walk('.'):
#     # Examine each file
#     for file in filenames:
#         # If it's a text file...
#         if os.path.splitext(file)[1] == ('.txt'):
#             # Grab the full file path
#             full_name = os.path.join(root,file)
#             # For files ending in "ham.txt"
#             if fnmatch.fnmatch(file, '*ham.txt'):
#                 # print("Matched ham: "+ str(full_name))
#                 # Open the file and ignore any characters that aren't part of the utf-8 character set (emojis, etc.)
#                 with open(full_name, 'r', encoding='utf-8', errors='ignore') as current_file:
#                     data = current_file.read()
#                     # Strip out newlines so that the file is all one long string
#                     data = data.replace('\n','')
#                     # Write this info to the all emails text file
#                     f.write(str(file_number) + "\t" + "ham" + "\t" + data + "\n")
#                     file_number += 1
#             # Same thing for spam, just change the label for the email to spam
#             elif fnmatch.fnmatch(file, '*spam.txt'):
#                 # print("Matched spam: "+ str(full_name))
#                 with open(full_name, 'r', encoding='utf-8', errors='ignore') as current_file:
#                     data = current_file.read()
#                     data = data.replace('\n','')
#                     f.write(str(file_number) + "\t" + "spam" + "\t" + data + "\n")
#                     file_number += 1

aggregated = pd.read_csv('./all_emails.txt', sep = '\t', names = ['message_id','labels','message'])
print(aggregated.head())
print()
print(aggregated.describe())
print()
print(aggregated.groupby('labels').describe())
# print(aggregated['message_id'])
# print(aggregated['labels'])
print(aggregated['message'])

aggregated['length'] =(aggregated['message'].str.len())
print(aggregated.head())
print()
aggregated['length'].plot(bins=100,kind='hist')
plt.savefig('email_length_histogram.png', dpi = 200)
print(aggregated.length.describe())

def text_process(mess):
    nopunc = ''
    try:
        nopunc =[char for char in mess if char not in string.punctuation]
        nopunc=''.join(nopunc)
    except:
        pass
    return [word for word in nopunc.split()]
# return [word for word in nopunc.split() if word.lower() not in stopwords.words('english')]

print(aggregated['message'].head(5).apply(text_process))


## Machine Learning
# Take all of the emails as an aggregate and count the unique words found
bow_transformer = CountVectorizer(analyzer=text_process).fit(aggregated['message'])
# Print out how many unique words there are
print(len(bow_transformer.vocabulary_))

# Grab the 4th email in the dataset and print out the message
message4=aggregated['message'][3]
print(message4)
# Convert it to a quantitative state.
bow4=bow_transformer.transform([message4])
# Print the message in the transformed state
# For example, (0,36) means the 37th word from the dataset occurs in this email
# The row number is always zero because it's a one-dimensional list
print(bow4)
# Every value in this matrix is either a 1 (word occurs) or 0 (word does not occur)
print(bow4.shape)

messages_bow = bow_transformer.transform(aggregated['message'])

# Indicates the number of rows (messages) and columns (unique words)
# Call it sparse because most entries are zero
# since most emails will not have a large subset of all words found in the
# entire database.
print('Shape of Sparse Matrix: ',messages_bow.shape)
# Tells the total number of non-zero entries in the sparse matrix
print('Amount of non-zero occurences:',messages_bow.nnz)
# Measures what percent of the entries are populated with non-zero values.
sparsity =(100.0 * messages_bow.nnz/(messages_bow.shape[0]*messages_bow.shape[1]))
print('sparsity:{}'.format(round(sparsity,4)))

# Calculate the IDF value for each unique word in this dataset
tfidf_transformer=TfidfTransformer().fit(messages_bow)
tfidf4 = tfidf_transformer.transform(bow4)
print(tfidf4)

messages_tfidf=tfidf_transformer.transform(messages_bow)
print(messages_tfidf.shape)

# Feed the IDF values into a machine learning model. Use the words from each email to create a
# # model that can predict whether a message is spam or ham.
# Print out an example of a predicted label as well as an actual label.
spam_detect_model = MultinomialNB().fit(messages_tfidf,aggregated['labels'].astype(str))
print('predicted:',spam_detect_model.predict(tfidf4)[0])
print('expected:',aggregated['labels'][3])

# Use the model above to make predictions for all emails in the dataset.
all_predictions = spam_detect_model.predict(messages_tfidf)
print(all_predictions)

# Print out a report indicating the precision, recall, f1-score and support
# for each message type.
print(classification_report(aggregated['labels'].astype(str),all_predictions, digits = 4))

# Print out a matrix of the raw counts used to create the classification report above.
# Numbers on the main diagonal indicate correct predictions.
# Numbers elsewhere indicate incorrect predictions.
print(confusion_matrix(aggregated['labels'].astype(str),all_predictions))
