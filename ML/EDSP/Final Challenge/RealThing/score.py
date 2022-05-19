
def init():
    global model
    # Replace filename if needed.
    path = os.getenv('AZUREML_MODEL_DIR') 
    #model_path = os.path.join(path, 'EmployeeLeftXYZ_model.pkl')
    model_path = filename
    # Deserialize the model file back into a sklearn model.
    model = joblib.load(model_path)

    
input_sample = pd.DataFrame(data=[{    
    "IT Touchpoints": 2,
    "behaviorPattern3": 2,
    "behaviorPattern4": 5,
    "behaviorPattern5":4,
    "fraudTraining Completed":1,
    "peerUsageMetric4":3,
    "peerUsageMetric6":0,
    "usageMetric2": 2,
    "usageMetric3":0,
    "usageMetric5":2
}])


# This is an integer type sample. Use the data type that reflects the expected result.
output_sample = np.array([0])

# To indicate that we support a variable length of data input,
# set enforce_shape=False
@input_schema('data', PandasParameterType(input_sample))
@output_schema(NumpyParameterType(output_sample))
def run(data):
    try:
        print("input_data....")
        print(data.columns)
        print(type(data))
        result = model.predict(data)
        print("result.....")
        print(result)
    # You can return any data type, as long as it can be serialized by JSON.
        return result.tolist()
    except Exception as e:
        error = str(e)
        return error
