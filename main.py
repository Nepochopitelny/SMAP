import pandas as pd

sms_spam = pd.read_csv('spam_NStugard.csv', sep=',',
header=None, names=['Label', 'SMS'])

print(sms_spam.shape)
sms_spam.head()
sms_spam['Label'].value_counts(normalize=True)
# Randomize the dataset
data_randomized = sms_spam.sample(frac=1, random_state=1)

# Calculate index for split
training_test_index = round(len(data_randomized) * 0.8)

# Split into training and test sets
training_set = data_randomized[:training_test_index].reset_index(drop=True)
test_set = data_randomized[training_test_index:].reset_index(drop=True)

print(training_set.shape)
print(test_set.shape)

print(training_set['Label'].value_counts(normalize=True))
print(test_set['Label'].value_counts(normalize=True))

print("Before cleaning")
print(training_set.head(3))
print("#############################")
print("After cleaning")
training_set['SMS'] = training_set['SMS'].str.replace('\W', ' ')
training_set['SMS'] = training_set['SMS'].str.lower()
print(training_set.head(3))
