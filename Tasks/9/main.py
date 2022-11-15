import plotly.express as px

from sklearn.datasets import load_iris

data = load_iris(as_frame = True) #Load dataset
predictors = data.data #Prediction parameters
target = data.target #Iris types
target_names = data.target_names #Iris names

fig = px.histogram(predictors, target)
fig.update_layout(bargap=0.2)
fig.show()
#print(data)
# print(predictors.head(5))
# print('Target variable:\n', target.head(5))
#print('Names:\n', target_names)
