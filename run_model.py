import re
import os
import tensorflow as tf
from keras.models import model_from_json
import pandas as pd
import numpy as np


from nltk.corpus import stopwords
from sklearn.model_selection import train_test_split
from sklearn.metrics import median_absolute_error as mae
from sklearn.metrics import mean_squared_error as mse
from sklearn.metrics import accuracy_score as acc
import matplotlib.pyplot as plt

from keras.models import Sequential
from keras import initializers
from keras.layers import Dropout, Activation, Embedding, Convolution1D, MaxPooling1D, Input, Dense, BatchNormalization, Flatten, Reshape, Concatenate, Add
from keras.layers.recurrent import LSTM, GRU
from keras.callbacks import Callback, ModelCheckpoint, EarlyStopping, ReduceLROnPlateau
from keras.models import Model
from keras.optimizers import Adam, SGD, RMSprop
from keras import regularizers
import pickle

def unnormalize(price):
    '''Revert values to their unnormalized amounts'''
    #price = (price+1)/2
    price = price*(926.549805000001+768.040039)-768.040039
    return(price)

def normalize(price):
    '''Revert values to their unnormalized amounts'''
    price = (price+768.040039)/(926.549805000001+768.040039)
    return(price)

with open("pad_headlines_1.txt", "rb") as fp:   # Unpickling
     pad_headlines = pickle.load(fp)
with open("norm_price_1.txt", "rb") as fp:   # Unpickling
     norm_price = pickle.load(fp)
with open("price_his_checked.txt", "rb") as fp:   # Unpickling
     price_his = pickle.load(fp)
with open("vol_his.txt", "rb") as fp:   # Unpickling
     vol_his = pickle.load(fp)
with open("word_embedding_matrix_1.txt", "rb") as fp:   # Unpickling
     word_embedding_matrix = pickle.load(fp)
embedding_dim = 300
nb_words = len(word_embedding_matrix)
max_headline_length = 16
max_daily_length = 200
vol_price=[]

for i in range(len(vol_his)):
	vol_price1=[]
	for k in range(49):
		vol_price1.append([price_his[i][k],vol_his[i][k]])
	vol_price.append(vol_price1)
#norm_price = nor
norm_price.reverse()
x_train, x_test, y_train, y_test = train_test_split(pad_headlines, norm_price, test_size = 0.15, random_state = 2, shuffle=False)

x_train = np.array(x_train)
x_test = np.array(x_test)
y_train = np.array(y_train)
y_test = np.array(y_test)



# In[308]:

# Check the lengths




# In[310]:

filter_length1 = 3
filter_length2 = 5
dropout = 0.5
learning_rate = 0.001
weights = initializers.TruncatedNormal(mean=0.0, stddev=0.1, seed=2)
nb_filter = 64
rnn_output_size = 128
hidden_dims = 256

deeper = False
wider = False
dropout=0.2
learning_Rate = 0.0001

if wider == True:
    nb_filter *= 2
    rnn_output_size *= 2
    hidden_dims *= 2


def build_model():
    
    model1 = Sequential()
    
    model1.add(Embedding(nb_words, 
                         embedding_dim,
                         weights=[word_embedding_matrix], 
                         input_length=max_daily_length))
    model1.add(Dropout(dropout))
    
    model1.add(Convolution1D(filters = nb_filter, 
                             kernel_size = filter_length1, 
                             padding = 'same',
                            activation = 'relu'))
    model1.add(Dropout(dropout))

    #model1.add(MaxPooling1D(pool_length=1))
    model1.add(LSTM(rnn_output_size, 
                   #activation=None,
                   kernel_initializer=weights,
                   dropout = dropout,
                   return_sequences=True))

    model1.add(LSTM(rnn_output_size, 
                   #activation=None,
                   kernel_initializer=weights,
                   dropout = dropout))
    
    ####
    
    model2 = Sequential()
    
    model2.add(Embedding(nb_words, 
                         embedding_dim,
                         weights=[word_embedding_matrix], 
                         input_length=max_daily_length))
    model2.add(Dropout(dropout))
    
    
    model2.add(Convolution1D(filters = nb_filter, 
                             kernel_size = filter_length2, 
                             padding = 'same',
                             activation = 'relu'))
    #model2.add(MaxPooling1D(pool_length=1))
    model2.add(Dropout(dropout))
    
    model2.add(LSTM(rnn_output_size, 
                    #activation=None,
                    kernel_initializer=weights,
                    dropout = dropout,
                    return_sequences=True))
    
    model2.add(LSTM(rnn_output_size, 
                   #activation=None,
                   kernel_initializer=weights,
                   dropout = dropout))
    
    ####

    #model12 = Sequential()
    #model12 = Add()([model1.output, model2.output])

    '''
    model3 = Sequential()
    model3.add(LSTM(rnn_output_size, input_shape=(49,2), kernel_initializer=weights, return_sequences=True))
    model3.add(Dropout(dropout))

    model3.add(LSTM(128, kernel_initializer=weights, return_sequences=True))
    model3.add(LSTM(128, kernel_initializer=weights, return_sequences=False))
    model3.add(Dropout(dropout))
	'''
    model = Add()([model1.output, model2.output])
    
    model = Dense(hidden_dims, activation = 'relu', kernel_initializer = weights)(model)
    #model = Dense(256, kernel_initializer=weights)(model)
    #model = Dense(128, kernel_initializer=weights)(model)
    model = Dense(1, kernel_initializer=weights)(model)

    new_model = Model([model1.input, model2.input], model)
    new_model.compile(loss='mean_squared_error', optimizer=Adam(lr=learning_rate, clipvalue=1.0))
    print(new_model.summary())
    return new_model
    '''
    model.add(Concatenate([model1, model2]))
    
    model.add(Dense(hidden_dims, kernel_initializer=weights))
    model.add(Dropout(dropout))
    
    if deeper == True:
        model.add(Dense(hidden_dims//2, kernel_initializer=weights))
        model.add(Dropout(dropout))

    model.add(Dense(1, 
                    kernel_initializer = weights,
                    name='output'))

    model.compile(loss='mean_squared_error',
                  optimizer=Adam(lr=learning_rate,clipvalue=1.0))
    return model
    '''


# In[311]:
'''
# Use grid search to help find a better model
for deeper in [False]:
    for wider in [True,False]:
        for learning_rate in [0.001]:
            for dropout in [0.3, 0.5]:
                model = build_model()
                print()
                print("Current model: Deeper={}, Wider={}, LR={}, Dropout={}".format(
                    deeper,wider,learning_rate,dropout))
                print()
                save_best_weights = 'question_pairs_weights_deeper={}_wider={}_lr={}_dropout={}.h5'.format(
                    deeper,wider,learning_rate,dropout)

                callbacks = [ModelCheckpoint(save_best_weights, monitor='val_loss', save_best_only=True),
                             EarlyStopping(monitor='val_loss', patience=5, verbose=1, mode='auto'),
                             ReduceLROnPlateau(monitor='val_loss', factor=0.2, verbose=1, patience=3)]

                history = model.fit([x_train,x_train],
                                    y_train,
                                    batch_size=128,
                                    epochs=100,
                                    validation_split=0.15,
                                    verbose=True,
                                    shuffle=True,
                                    callbacks = callbacks)

'''
# In[312]:

# Make predictions with the best weights
print("Train start:")

# Need to rebuild model in case it is different from the model that was trained most recently.
model = build_model()
model.fit([x_train, x_train], y_train, batch_size=16, epochs=15, validation_split=0.15, verbose=True, shuffle=True)
#model.load_weights('./question_pairs_weights_deeper={}_wider={}_lr={}_dropout={}.h5'.format(
#                    deeper,wider,learning_rate,dropout))
predictions = model.predict([x_test,x_test], verbose = True)

print("Train finished!")

unnorm_predictions = []
for pred in predictions:
    unnorm_predictions.append(unnormalize(pred))
unnorm_y_test = []
for y in y_test:
    unnorm_y_test.append(unnormalize(y))

# Compare testing loss to training and validating loss
print("Mse:")
print(mse(unnorm_y_test, unnorm_predictions))
'''
json_file = open('model.json', 'r')
loaded_model_json = json_file.read()
json_file.close()
loaded_model = model_from_json(loaded_model_json)
loaded_model.compile(loss='mean_squared_error', optimizer=Adam(lr=learning_rate, clipvalue=1.0))
'''
# In[314]:

# In[346]:

# Calculate the median absolute error for the predictions
print(mae(unnorm_y_test, unnorm_predictions))


# In[362]:

print("Summary of actual opening price changes")
print(pd.DataFrame(unnorm_y_test, columns=[""]).describe())
print()
print("Summary of predicted opening price changes")
print(pd.DataFrame(unnorm_predictions, columns=[""]).describe())


# In[365]:

# Plot the predicted (blue) and actual (green) values
plt.figure(figsize=(12,4))
plt.plot(unnorm_predictions)
plt.plot(unnorm_y_test)
plt.title("Predicted (blue) vs Actual (green) Opening Price Changes")
plt.xlabel("Testing instances")
plt.ylabel("Change in Opening Price")
plt.show()




# Create lists to measure if opening price increased or decreased
direction_pred = []
for pred in unnorm_predictions:
    if pred >= 0:
        direction_pred.append(1)
    else:
        direction_pred.append(0)
direction_test = []
for value in unnorm_y_test:
    if value >= 0:
        direction_test.append(1)
    else:
        direction_test.append(0)


# In[367]:

# Calculate if the predicted direction matched the actual direction
direction = acc(direction_test, direction_pred)
direction = round(direction,4)*100
print("Predicted values matched the actual direction {}% of the time.".format(direction))

with open("unnorm_predictions.txt", "wb") as fp:   #Pickling
      pickle.dump(unnorm_predictions, fp)

with open("unnorm_y_test.txt", "wb") as fp:   #Pickling
      pickle.dump(unnorm_y_test, fp)

# As we can see from the data above, this model struggles to accurately predict the change in the opening price of the Dow Jones Instrustial Average. Given that its median average error is 74.15 and the interquartile range of the actual price change is 142.16 (87.47 + 54.69), this model is about as good as someone who knows the average price change of the Dow. 
# 
# I have a few ideas for why this model struggles:
# - The market is arguably to be a random walk. Although there is some direction to its movements, there is still quite a bit of randomness to its movements.
# - The news that we used might not be the most relevant. Perhaps it would have been better to use news relating to the 30 companies that make up the Dow.
# - More information could have been included in the model, such as the previous day(s)'s change, the previous day(s)'s main headline(s). 
# - Many people have worked on this task for years and companies spend millions of dollars to try to predict the movements of the market, so we shouldn't expect anything too great considering the small amount of data that we are working with and the simplicity of our model.

# ## Make Your Own Predictions

# Below is the code necessary to make your own predictions. I found that the predictions are most accurate when there is no padding included in the input data. In the create_news variable, I have some default news that you can use, which is from April 30th, 2017. Just change the text to whatever you want, then see the impact your new headline will have.

# In[117]:

def news_to_int(news):
    '''Convert your created news into integers'''
    ints = []
    for word in news.split():
        if word in vocab_to_int:
            ints.append(vocab_to_int[word])
        else:
            ints.append(vocab_to_int['<UNK>'])
    return ints


# In[118]:

def padding_news(news):
    '''Adjusts the length of your created news to fit the model's input values.'''
    padded_news = news
    if len(padded_news) < max_daily_length:
        for i in range(max_daily_length-len(padded_news)):
            padded_news.append(vocab_to_int["<PAD>"])
    elif len(padded_news) > max_daily_length:
        padded_news = padded_news[:max_daily_length]
    return padded_news


# In[368]:
'''
# Default news that you can use
create_news = "Leaked document reveals Facebook conducted research to target emotionally vulnerable and insecure youth.                Woman says note from Chinese 'prisoner' was hidden in new purse.                21,000 AT&T workers poised for Monday strike                housands march against Trump climate policies in D.C., across USA                Kentucky judge won't hear gay adoptions because it's not in the child's \"best interest\"                Multiple victims shot in UTC area apartment complex                Drones Lead Police to Illegal Dumping in Riverside County | NBC Southern California                An 86-year-old Californian woman has died trying to fight a man who was allegedly sexually assaulting her 61-year-old friend.                Fyre Festival Named in $5Million+ Lawsuit after Stranding Festival-Goers on Island with Little Food, No Security.                The \"Greatest Show on Earth\" folds its tent for good                U.S.-led fight on ISIS have killed 352 civilians: Pentagon                Woman offers undercover officer sex for $25 and some Chicken McNuggets                Ohio bridge refuses to fall down after three implosion attempts                Jersey Shore MIT grad dies in prank falling from library dome                New York graffiti artists claim McDonald's stole work for latest burger campaign                SpaceX to launch secretive satellite for U.S. intelligence agency                Severe Storms Leave a Trail of Death and Destruction Through the U.S.                Hamas thanks N. Korea for its support against ‘Israeli occupation’                Baker Police officer arrested for allegedly covering up details in shots fired investigation                Miami doctor’s call to broker during baby’s delivery leads to $33.8 million judgment                Minnesota man gets 15 years for shooting 5 Black Lives Matter protesters                South Australian woman facing possible 25 years in Colombian prison for drug trafficking                The Latest: Deal reached on funding government through Sept.                Russia flaunts Arctic expansion with new military bases"

clean_news = clean_text(create_news)

int_news = news_to_int(clean_news)

pad_news = padding_news(int_news)

pad_news = np.array(pad_news).reshape((1,-1))

pred = model.predict([pad_news,pad_news])

price_change = unnormalize(pred)

print("The Dow should open: {} from the previous open.".format(np.round(price_change[0][0],2)))
'''

