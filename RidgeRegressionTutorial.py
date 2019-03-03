import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import Ridge
from sklearn.linear_model import RidgeCV
import random


''' Problem : We will generate a data-set containing the information regarding Squirrels. The information inclues size, weight and the intake of the squirrels
in grams. We need to use Ridge Regression to predict the size of the a mouse based on the other independent variables.
We will be implementing Ridge Regression using the linear Ridge model and using Cross-validation model.
We will be generating the values of the data-set ourselves and we will limit the dataset to minimum values as that is the whole
point of applying Ridge Regression Model, to predict efficiently even when there is limited data by using a Ridge Regression Penalty to
approximate the best fitting mapping function.

The linear regression equation can be initially considered to be :

SIZE = CONSTANT + 13.0*WEIGHT - 0.8*INTAKE

where CONSTANT = 12
'''

#Intializing parameters

CONSTANT = 12               # Also known as the Y-Intercept
TOTAL_SAMPLES = 200         # Total samples that are going to be used
MAX_SAMPLE_SIZE = 40        # Maximum size of a particular squirrel
INVOKE_ERROR = 0.02         # Error/Deviation that must be randomly applied to the data generated


#Defining functions to generate values for 13*WEIGHT and -0.8*INTAKE

def generateWeight(x):
    result = (x*13.0)
    deviation = 0
    deviation = result * random.uniform(-INVOKE_ERROR,INVOKE_ERROR)
    return(result + deviation)

def generateIntake(y):
    result = (y*(-0.80))
    deviation = 0
    deviation = result * random.uniform(-INVOKE_ERROR,INVOKE_ERROR)
    return (result + deviation)

#Specifying how data will be divided into Training and Testing (Training = 40%, Testing = 60%)

train_data_count = (int(TOTAL_SAMPLES*40/100))
test_data = TOTAL_SAMPLES - train_data_count


#Generating values
random_array = np.random.randint(MAX_SAMPLE_SIZE,size=TOTAL_SAMPLES).astype(float)
weight = generateWeight(random_array)
intake = generateIntake(random_array)
rhs = weight + intake             #RHS of the given Equation
size = rhs + CONSTANT              #SIZE = RHS + Y-Intercept ===> SIZE = CONSTANT + 13.0*WEIGHT - 0.8*INTAKE
                                    #SIZES contains an array of the randomly generated values of the equation, as done in
                                    #Linear-Regression to fit a line.

#Dividing the generated dataset into Training and Testing

weight_train, weight_test = np.split(weight,[train_data_count,])
intake_train, intake_test = np.split(intake,[train_data_count,])
size_train, size_test = np.split(size,[train_data_count,])


#Creating a dictionary so that we can convert the data into Pandas Dataframe

independent_training_data = {'WEIGHT':weight_train,     #Independent Variables Training Data (WEIGHT AND INTAKE)
              'INTAKE':intake_train}

dependent_training_data = {'SIZE':size_train}           #Dependent Variables Training Data (SIZE)

independent_test_data = {'WEIGHT':weight_test,       #Independent Variables Testing Data (WEIGHT AND INTAKE)
              'INTAKE':intake_test}

dependent_test_data = {'SIZE':size_test}             #Dependent Variables Testing Data (SIZE)

#Generating Pandas Dataframe

independent_training_dataframe = pd.DataFrame(data=independent_training_data)
independent_testing_dataframe = pd.DataFrame(data=independent_test_data)
dependent_training_dataframe = pd.DataFrame(data=dependent_training_data)
dependent_testing_dataframe = pd.DataFrame(data=dependent_test_data)


#Initiliazing the Ridge Regression Model (Without Cross Validation)

model = Ridge(alpha=0.7)
model.fit(independent_training_dataframe,dependent_training_dataframe)

#Predicting values using Model with Cross Validation

print("\nRidge-Regression model without cross validation predictions:",end="\n")
predictions = model.predict(independent_testing_dataframe)
print("Prediction from a random value:",model.predict([[random.uniform(weight_train.min(),weight_train.max()),random.uniform(intake_train.min(),intake_train.max())]]))
print("Model score:",model.score(independent_testing_dataframe,dependent_testing_dataframe))
print("Co-efficients for line fitted by model after regression penalty:")
for coeff in model.coef_:
    print(coeff)
print("Y-intercept for the model:",model.intercept_)
print("\n\n")

#Plotting the datapoints on a graph

difference = np.subtract(dependent_testing_dataframe,predictions)

plt.title("Ridge Regression Comparison")
plt.xlabel("WEIGHT AND INTAKE")
plt.ylabel("SIZE")
plt.plot(np.arange(difference.size),difference,color="red")
plt.show()















