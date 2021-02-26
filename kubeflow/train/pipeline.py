
from main import main

import kfp
import kfp.dsl as dsl
import kfp.components as comp

def pipeline():
    # Define the pipeline
    # @dsl.pipeline is a required decoration including the name and description properties.
    @kfp.dsl.pipeline(
    name='Load forecast Pipeline',
    description='A pipeline that performs model training and prediction.'
    )

    def training_pipeline(gcp_bucket: str, project: str):
            # pre_image = f"gcr.io/{project}/pre_image:{github_sha}"
            # train_forecast_image = f"gcr.io/{project}/train_forecast_image:{github_sha}"
            operations = {}
            operations['training'] = dsl.ContainerOp(
                name='Training',
                image= 'rajatsethi7/my_docker_image',
                command=['python3'],
                arguments=["main.py"]
            )
            

            dsl.get_pipeline_conf()

            return operations
        
    return training_pipeline   








# Define parameters to be fed into pipeline
def training_container_pipeline(
    data_path: str,
    model_file: str,
    test_days: int,
    proposal_id: str,
    ):
    
    # Create training component.
    load_forecast_training_container = train_op(model_file, test_days, proposal_id)


# API Client for KubeFlow Pipeline.
client = kfp.Client()

pipeline_func = training_container_pipeline
#Create an experiment to associate with a pipeline run
experiment_name = 'training_kubeflow'
run_name = pipeline_func.__name__ + ' run'

arguments = {"model_file": model_file,
             "test_days": test_days,
             "proposal_id": proposal_id,
            }

# Compile pipeline to generate compressed YAML definition of the pipeline. 
# The DSL compiler transforms this pipeline’s Python code into a static configuration (YAML).
kfp.compiler.Compiler().compile(pipeline_func, '{}.zip'.format(experiment_name))


