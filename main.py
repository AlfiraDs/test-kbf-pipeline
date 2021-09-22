import kfp
import kfp.dsl as dsl
import kfp.components as comp


def train(data_path, model_file):
    import pickle
    from sklearn.ensemble import RandomForestClassifier
    from sklearn.metrics import roc_auc_score
    from sklearn.datasets import load_wine
    from sklearn.model_selection import train_test_split

    X, y = load_wine(return_X_y=True)
    y = y == 2
    X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=42)
    model = RandomForestClassifier(n_estimators=10, random_state=42)
    model.fit(X_train, y_train)

    # Evaluate the model and print the results
    test_roc_auc = roc_auc_score(y_test, model.predict(X_test))
    print('Test roc_auc:', test_roc_auc)

    # Save the model to the designated
    with open(f'{data_path}/{model_file}', 'wb') as f:
        pickle.dump(model, f)

    # Save the test_data as a pickle file to be used by the predict component.
    with open(f'{data_path}/test_data', 'wb') as f:
        pickle.dump((X_test, y_test), f)


def predict(data_path, model_file, sample_number):
    import pickle
    import numpy as np

    with open(f'{data_path}/{model_file}', 'rb') as f:
        model = pickle.load(f)

    with open(f'{data_path}/test_data', 'rb') as f:
        X_test, y_test = pickle.load(f)

    sample_number = int(sample_number)
    img = X_test[sample_number]
    img = (np.expand_dims(img, 0))
    predictions = model.predict(img)
    prediction = np.argmax(predictions[0])
    true_label = y_test[sample_number]

    with open(f'{data_path}/result.txt', 'w') as result:
        result.write(f"Prediction: {prediction} | Actual: {true_label}")

    print('Prediction has be saved successfully!')


# train(r'./data', 'model.pkl')
# predict(r'./data', 'model.pkl', 0)


# Create train and predict lightweight components.
train_op = comp.func_to_container_op(train, base_image='ecoron/python36-sklearn')
predict_op = comp.func_to_container_op(predict, base_image='ecoron/python36-sklearn')

# Create a client to enable communication with the Pipelines API server.
# client = kfp.Client(host='pipelines-api.kubeflow.svc.cluster.local:8888')
# client = kfp.Client(host='http://kubeflow01.sfo.corp.globant.com/_/pipeline/?ns=aliaksandr-lashkov')
# client = kfp.Client(host='http://kubeflow01.sfo.corp.globant.com/_/pipeline/#/pipelines:3000')
client = kfp.Client(host='http://kubeflow01.sfo.corp.globant.com/pipeline')


# Define the pipeline
@dsl.pipeline(
    name='WINE Pipeline',
    description='A toy pipeline that performs WINE model training and prediction.'
)
# Define parameters to be fed into pipeline
def wine_container_pipeline(
        data_path: str,
        model_file: str,
        sample_number: int
):
    # Define volume to share data between components.
    vop = dsl.VolumeOp(
        name="create_volume",
        resource_name="data-volume",
        size="1Gi",
        modes=dsl.VOLUME_MODE_RWM)

    wine_training_container = train_op(data_path, model_file).add_pvolumes({data_path: vop.volume})
    wine_predict_container = predict_op(data_path, model_file, sample_number).add_pvolumes(
        {data_path: wine_training_container.pvolume})

    wine_result_container = dsl.ContainerOp(
        name="print_prediction",
        image='library/bash:4.4.23',
        pvolumes={data_path: wine_predict_container.pvolume},
        arguments=['cat', f'{data_path}/result.txt']
    )


def main():
    DATA_PATH = '/mnt'
    MODEL_PATH = 'wine_model.pkl'
    SAMPLE_NUMBER = 0

    pipeline_func = wine_container_pipeline
    experiment_name = 'wine_kubeflow'
    run_name = pipeline_func.__name__ + ' run'

    arguments = {"data_path": DATA_PATH,
                 "model_file": MODEL_PATH,
                 "image_number": SAMPLE_NUMBER}

    # Compile pipeline to generate compressed YAML definition of the pipeline.
    kfp.compiler.Compiler().compile(pipeline_func,
                                    '{}.zip'.format(experiment_name))

    # # Submit pipeline directly from pipeline function
    # run_result = client.create_run_from_pipeline_func(pipeline_func=pipeline_func,
    #                                                   arguments=arguments,
    #                                                   run_name=run_name,
    #                                                   experiment_name=experiment_name,
    #                                                   pipeline_conf=None,
    #                                                   namespace='alex-test-namespace'
    #                                                   )
    response = client.upload_pipeline(
        pipeline_package_path='{}.zip'.format(experiment_name),
        pipeline_name=experiment_name,
        description=f'{experiment_name} description',
    )
    print(f'uploading pipeline response: \n{response}')

    response = client.upload_pipeline_version(
        pipeline_package_path='{}.zip'.format(experiment_name),
        pipeline_version_name=experiment_name + '_v1',
        pipeline_id=None,
        pipeline_name=experiment_name)
    print(f'uploading pipeline version response: \n{response}')


if __name__ == '__main__':
    main()
