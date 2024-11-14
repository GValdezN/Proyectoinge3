from flask import Flask, jsonify, request, render_template
import numpy as np
import boto3
from PIL import Image
import io
import json

# Clases de predicción
classes = ['cardboard', 'glass', 'metal', 'organic', 'paper', 'plastic', 'trash']

app = Flask(__name__)

# Configuración del cliente de SageMaker
SAGEMAKER_ENDPOINT_1 = 'Densenet121'  # Endpoint para Densenet121.h5
SAGEMAKER_ENDPOINT_2 = 'Densenet121BaseModel'  # Endpoint para Densenet121_BaseModel.h5
sagemaker_client = boto3.client('runtime.sagemaker', region_name='us-east-1')

def model_predict(data, endpoint_name):
    try:
        # Preprocesar la imagen para enviarla al endpoint
        image_res = data.resize((224, 224), Image.BILINEAR)
        image_dim = np.array(image_res)
        image_dim = np.expand_dims(image_dim, axis=0)
        payload = json.dumps(image_dim.tolist())

        # Realizar la invocación al endpoint de SageMaker
        response = sagemaker_client.invoke_endpoint(
            EndpointName=endpoint_name,
            ContentType='application/json',
            Body=payload
        )

        # Procesar la respuesta de SageMaker
        result = json.loads(response['Body'].read().decode())
        predicted_class = classes[np.argmax(result)]
        confidence_percentage = str(round(result[0][np.argmax(result)] * 100, 2))
        
        return predicted_class, confidence_percentage
    except Exception as e:
        print(f"Error al invocar SageMaker: {e}")
        return "Error", "0"

@app.route('/', methods=['GET'])
def home():
    return render_template('index.html')  

@app.route('/predict', methods=['POST'])
def upload():
    if request.method == 'POST':
        try:
            # Obtiene el archivo del request
            f = request.files['file']
            # Convierte el archivo en una imagen de PIL
            image = Image.open(io.BytesIO(f.read()))

            # Llamar a ambos modelos
            predicted_class_1, confidence_percentage_1 = model_predict(image, SAGEMAKER_ENDPOINT_1)
            predicted_class_2, confidence_percentage_2 = model_predict(image, SAGEMAKER_ENDPOINT_2)

            # Preparar la respuesta JSON combinada
            result = {
                "model_1": {
                    "predicted_class": predicted_class_1,
                    "confidence_percentage": confidence_percentage_1
                },
                "model_2": {
                    "predicted_class": predicted_class_2,
                    "confidence_percentage": confidence_percentage_2
                }
            }
            
            return jsonify(result)
        except Exception as e:
            print(f"Error en la predicción: {e}")
            return jsonify({"error": "Hubo un problema en la predicción. Inténtalo de nuevo."}), 500
    return None

if __name__ == '__main__':
     app.run(host="0.0.0.0", port=5000)
