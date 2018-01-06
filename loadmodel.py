from keras.models import Sequential, model_from_json

model = model_from_json(open('DRIVE_Logistic_model.json').read())# if json 
# model = model_from_yaml(open('my_model_architecture.yaml').read())# if yaml 
model.load_weights('DRIVE_Logistic_wts.h5')

print(model.metrics_names)