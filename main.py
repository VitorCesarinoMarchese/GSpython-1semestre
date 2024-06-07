import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression

def TemperaturePredictor(temperatures, future_steps):
    X = np.arange(2010, 2010 + len(temperatures)).reshape(-1, 1)
    y = np.array(temperatures)
    
    model = LinearRegression()
    model.fit(X, y)
    
    future_X = np.arange(2010+len(temperatures), 2010+len(temperatures) + future_steps).reshape(-1, 1)
    future_y = model.predict(future_X)
    
    plt.figure(figsize=(15, 7))
    plt.plot(X, y, label='Today Temperatures', color="red")
    plt.scatter(X, y, color='red')
    plt.plot(future_X, future_y, 'r--', label='Expected Temperatures')
    plt.xlabel('Time')
    plt.ylabel('Temperatures')
    plt.title('Temperature Predictions')
    plt.legend()
    plt.show()


def LuxPredictor(Lux, future_steps):
    X = np.arange(2010, 2010 + len(Lux)).reshape(-1, 1)
    y = np.array(Lux)

    model = LinearRegression()
    model.fit(X, y)

    future_X = np.arange(2010+len(Lux), 2010+len(Lux) + future_steps).reshape(-1, 1)
    future_y = model.predict(future_X)

    plt.figure(figsize=(15, 7))
    plt.plot(X, y, label='Today Lux', color="blue")
    plt.scatter(X, y, color='blue')
    plt.plot(future_X, future_y, 'r--', label='Expected Lux')
    plt.xlabel('Time')
    plt.ylabel('Lux')
    plt.title('Lux Predictions')
    plt.legend()
    plt.show()
def AcelerationPredictor(Acceleration, future_steps):
    X = np.arange(2010, 2010+len(Acceleration)).reshape(-1, 1)
    y = np.array(Acceleration)

    model = LinearRegression()
    model.fit(X, y)

    future_X = np.arange(2010+len(Acceleration), 2010+len(Acceleration) + future_steps).reshape(-1, 1)
    future_y = model.predict(future_X)

    plt.figure(figsize=(15, 7))
    plt.plot(X, y, label='Today Acceleration', color="green")
    plt.scatter(X, y, color='green')
    plt.plot(future_X, future_y, 'r--', label='Expected Acceleration')
    plt.xlabel('Time')
    plt.ylabel('Acceleration')
    plt.title('Acceleration Predictions')
    plt.legend()
    plt.show()
