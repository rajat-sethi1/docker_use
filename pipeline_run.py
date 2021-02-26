#!/usr/bin/env python3

from esap_forecast.kubeflow.train import main

import kfp
import kfp.dsl as dsl
import kfp.components as comp
    

@dsl.pipeline(
name='Load forecast Pipeline',
description='A pipeline that performs model training'
)

def training_pipeline():
    # Define the pipeline
    # @dsl.pipeline is a required decoration including the name and description properties.
    @kfp.dsl.pipeline(
    name='Load forecast Pipeline',
    description='A pipeline that performs model training and prediction.'
    )

    def train_pipeline():
        operations = {}
        operations['training'] = dsl.ContainerOp(
            name='Training',
            image= 'rajatsethi7/my_docker_image',
            command=['python3'],
            arguments=["main.py"]
        )
        
        dsl.get_pipeline_conf()

        return operations
        
    return train_pipeline   
