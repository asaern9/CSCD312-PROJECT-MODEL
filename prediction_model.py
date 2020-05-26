import tensorflow 
import numpy 

bc_model = tensorflow.keras.models.load_model('breast_cancer_model.h5')
prediction_data = 'prediction_data/file_name'

def model_prediction(IMG):
    img_path = IMG
    img = tensorflow.keras.preprocessing.image.load_img(img_path, target_size=(224,224))
    x = tensorflow.keras.preprocessing.image.img_to_array(img)
    x = numpy.expand_dims(x, axis=0)

    class_names = ['Benign', 'Malignant']

    predict_classes =  numpy.argmax(bc_model.predict(x), axis=-1)
    predict_accuracy = bc_model.predict(x)

    if predict_classes == 0:
        print('Test result:', class_names[0],'  Model Accuray:',numpy.max(predict_accuracy)*100)
    else:
        print('Test result:', class_names[1],'  Model Accuracy:',numpy.max(predict_accuracy)*100)


model_prediction(prediction_data)