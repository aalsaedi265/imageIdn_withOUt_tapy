
this project is going to have a very simple machine learning model which will training using scikit -learn to do the training of the model once that part is done will begin doing the restful api stuff from it. That will be done through fastapi but running the application that will ne uvicorn's job. addition's because i like to flex on my self will use pillow to be able to work with images.

## notes        

example how to create virtual Environment
py -m venv chat
.\chat\Scripts\activate
deactivate

pip3 install fastapi uvicorn pillow scikit-learn

uvicorn fileName:appName --reload

appName is the varable hold the comand which store the fastapi()


when you run the training you will get the mnist model to use