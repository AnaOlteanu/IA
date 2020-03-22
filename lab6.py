import numpy as np
from sklearn.linear_model import LinearRegression, Ridge
from sklearn.metrics import mean_absolute_error, mean_squared_error
from sklearn.utils import shuffle
from sklearn import preprocessing

training_data = np.load('data/training_data.npy')
prices = np.load('data/prices.npy')

# print('The first 4 samples are:\n ', training_data[:4])
# print('The first 4 prices are:\n ', prices[:4])
# shuffle
training_data, prices = shuffle(training_data, prices, random_state=0)


def normalize_data(train_data, test_data):
    scaler = preprocessing.StandardScaler()

    if scaler != None:
        scaler.fit(train_data)
        scaled_train_data = scaler.transform(train_data)
        scaled_test_data = scaler.transform(test_data)

        return scaled_train_data, scaled_test_data



num_samples_fold = len(training_data) // 3
#print(num_samples_fold)

# split train in 3 folds

training_data_1, prices_1 = training_data[:num_samples_fold], prices[:num_samples_fold]

training_data_2, prices_2 = training_data[num_samples_fold: 2 * num_samples_fold], prices[num_samples_fold: 2 * num_samples_fold]

training_data_3, prices_3 = training_data[2 * num_samples_fold:], prices[2 * num_samples_fold:]


def step(train_data, train_labels, test_data, test_labels,model):
    normalized_train_data, normalized_test_data = normalize_data(train_data, test_data)
    reg = model.fit(normalized_train_data, train_labels)
    mae = mean_absolute_error(test_labels, reg.predict(normalized_test_data))
    mse = mean_squared_error(test_labels, reg.predict(normalized_test_data))

    return mae, mse

model = LinearRegression()

#run 1
mae_1, mse_1 = step(np.concatenate((training_data_1, training_data_3)),np.concatenate((prices_1, prices_3)),
                    training_data_2,
                    prices_2,
                    model)
#run 2
mae_2, mse_2 = step(np.concatenate((training_data_1,training_data_2)),np.concatenate((prices_1,prices_2)),
                 training_data_3,
                 prices_3,
                 model)
#run 3
mae_3, mse_3 = step(np.concatenate((training_data_2,training_data_3)),np.concatenate((prices_2,prices_3)),
                  training_data_3,
                  prices_3,
                  model)

print("Mae 1:", mae_1)
print("Mae 2:", mae_2)
print("Mae 3:", mae_3)
print("------------------")
print("Mse 1:", mse_1)
print("Mse 2:", mse_2)
print("Mse 3:", mse_3)

mean_mae = (mae_1 + mae_2 + mae_3)/3
mean_mse = (mse_1 + mse_2 + mse_3)/3

print("------------------")
print("Overall mae:", mean_mae)
print("Overall mse:", mean_mse)
print("------------------")


for alpha_ in[1,10,100,1000]:
    model = Ridge(alpha = alpha_)
    print("alpha:",alpha_)


    #run 1
    mae_1, mse_1 = step(np.concatenate((training_data_1, training_data_3)), np.concatenate((prices_1, prices_3)),
                        training_data_2,
                        prices_2,
                        model)
    # run 2
    mae_2, mse_2 = step(np.concatenate((training_data_1, training_data_2)), np.concatenate((prices_1, prices_2)),
                        training_data_3,
                        prices_3,
                        model)
    # run 3
    mae_3, mse_3 = step(np.concatenate((training_data_2, training_data_3)), np.concatenate((prices_2, prices_3)),
                        training_data_3,
                        prices_3,
                        model)

    print("Mae 1:", mae_1)
    print("Mae 2:", mae_2)
    print("Mae 3:", mae_3)
    print("-------------------")
    print("Mse 1:", mse_1)
    print("Mse 2:", mse_2)
    print("Mse 3:", mse_3)

    mean_mae = (mae_1 + mae_2 + mae_3) / 3
    mean_mse = (mse_1 + mse_2 + mse_3) / 3

    print("--------------------")
    print("Overall mae:", mean_mae)
    print("Overall mse:", mean_mse)
    print(">>>>>>>>>>>>>>>>>>>>>")


model = Ridge(10)
scaler = preprocessing.StandardScaler()
scaler.fit(training_data)
normalized_train = scaler.transform(training_data)
model.fit(normalized_train,prices)

print("Coefficients for regression:",model.coef_)
print("Bias for regression:", model.intercept_)

features = ["Year","Kilometers Driven","Mileage","Engine","Power","Seats","Owner Type","Fuel Type","Transmission"]

index_max = np.argmax(np.abs(model.coef_))
index_min = np.argmin(np.abs(model.coef_))

most_significat_feature = features[int(index_max)]
second_most_significat_f = features[(index_max+1)]
least_significant_f = features[int(index_min)]

print("The most significant feature:", most_significat_feature)
print("The second most significant feature:", second_most_significat_f)
print("The least significant feature:", least_significant_f)
