import kfp
from kfp import dsl

def preprocess_op():

    return dsl.ContainerOp(
        name='Preprocess Data',
        image='rajatsethi7/my_docker_image',
        arguments=[],
        file_outputs={
            'x_train': '/app/x_train.npy',
            'x_test': '/app/x_test.npy',
            'y_train': '/app/y_train.npy',
            'y_test': '/app/y_test.npy',
        }
    )

@dsl.pipeline(
   name='Boston Housing Pipeline',
   description='An example pipeline that trains and logs a regression model.'
)
def boston_pipeline():
    _preprocess_op = preprocess_op()


if __name__ == '__main__':
    experiment_name = 'boston_price_kubeflow'
    kfp.compiler.Compiler().compile(boston_pipeline,'{}.zip'.format(experiment_name))
    client = kfp.Client()
    client.create_run_from_pipeline_func(boston_pipeline, arguments={})