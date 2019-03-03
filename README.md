# Machine-Learning-RidgeRegression

Problem : We will generate a data-set containing the information regarding Squirrels. The information inclues size, weight and the intake of the squirrels
in grams. We need to use Ridge Regression to predict the size of the a mouse based on the other independent variables.
We will be implementing Ridge Regression using the linear Ridge model and using Cross-validation model.
We will be generating the values of the data-set ourselves and we will limit the dataset to minimum values as that is the whole
point of applying Ridge Regression Model, to predict efficiently even when there is limited data by using a Ridge Regression Penalty to
approximate the best fitting mapping function.

The linear regression equation can be initially considered to be :

SIZE = CONSTANT + 13.0 * WEIGHT - 0.8 * INTAKE

where CONSTANT = 12

Dependencies used :

matplotlib - Used to plot the graph and show the regression line and the goodness of the fit.

pandas - Used to perform data operations and read information from the dataset.

numpy - General mathematical operations

Sklearn - To carry out operations on the data such as regression using the built-in functionalities.
