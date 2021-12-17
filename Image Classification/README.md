## Image Classification using AWS SageMaker
The goal of this project was to use AWS Sagemaker to train a pretrained model that can perform image classification by using the Sagemaker profiling, debugger, hyperparameter tuning and other good ML engineering practices.

### Project Set Up and Installation
 Instructions:
 - Enter AWS through the gateway in the course and open SageMaker Studio.
 - Download the starter files.
 - Download/Make the dataset available.

Pytorch deep learning framework was used in this project to build a CNN to classify images of dogs according to their breed. First, packages, like torch and smdebug were installed, and libraries including sagemaker, numpy, boto3, os, etc. were imported. 

### Dataset
The dog breed classification dataset provided by Udacity was used in the project. The dataset contains 8351 color images of dogs of 133 differnt breeds grouped into train, test and validation subsets in proportion 80:10:10 %. The images have different shapes and needed to be standardized by using transforms library. Several transforms including normalization were stacked in transforms.Compose.

#### Access
The data were uploaded to an S3 bucket through the AWS Gateway, so that SageMaker had access to the data. 

## Hyperparameter Tuning
Pretrained ResNet-50 model was used to provide transfer learning to the CNN. In order to obtain optimal accuracy values SageMaker provides utility for tuning different hyperparameters of the model. Learning rate and batch size hyperparameters were finetuned using the following ranges:
```
hyperparameter_ranges = {"lr": ContinuousParameter(0.001, 0.1),
					     "batch-size": CategoricalParameter([32, 64, 128, 256, 512]),}
```
The training results were logged, where Test Loss metric was provided to SageMaker with the "Minimize" objective.  

The following screenshots show completed training jobs and the hyperparameter values obtained from the best tuning job.

![training_hpo1](C:\Users\ikulk\OneDrive - ETEC Solutions\Documents\z_UDACITY\AWS_ML_nanodegree\nd009t-c3-deep-learning-topics-within-computer-vision-nlp-project-starter\training_hpo1.JPG)



![training_hpo2](C:\Users\ikulk\OneDrive - ETEC Solutions\Documents\z_UDACITY\AWS_ML_nanodegree\nd009t-c3-deep-learning-topics-within-computer-vision-nlp-project-starter\training_hpo2.JPG)



![training_hpo3](C:\Users\ikulk\OneDrive - ETEC Solutions\Documents\z_UDACITY\AWS_ML_nanodegree\nd009t-c3-deep-learning-topics-within-computer-vision-nlp-project-starter\training_hpo3.JPG)

## Debugging and Profiling
After obtaining the best hyperparameters a new model was created where SageMaker debugging and profiling utilities were used in order to track/detect potential issues with the model. In train_model.py script SMDebug hooks for PyTorch with the TRAIN and EVAL modes were added to the train and test functions correspondingly. In the main function SMDebug hook was created and registered to the model. The hook was also passed as an argument to the train and test functions.
Debugger rules and hook parameters were connfigured in the train_and_deploy notebook. With the help of SageMaker profiler instance resources like CPU and GPU memory utilization can be tracked. Profiler and debugger configuration were added to estimator to tarin the model.

### Results
Some issues were with Overtraining, PoorWeightInitialization, and LowGPUUtilization were found during training the model. Some CPU bottlenecks were encountered. A significant amount of time (79.98%) during training job was spent in phase "others" whereas most of the time should be spent on TRAIN and EVAL phases. Rules summary was provided in the profiling report. 

### Model Deployment
Originally the model was deployed as intended (by calling deploy on the Estimator), however querying the created endpoint resulted in the error presented on the following screenshot. I could not obtain any predictions on the inference because of the error on the screenshot below. 

![error](C:\Users\ikulk\OneDrive - ETEC Solutions\Documents\z_UDACITY\AWS_ML_nanodegree\nd009t-c3-deep-learning-topics-within-computer-vision-nlp-project-starter\error.JPG)

The new inference handler script was adapted from [here]( https://docs.aws.amazon.com/sagemaker/latest/dg/adapt-inference-container.html ), where the model_fn, input_fn and predict_fn added functionality for loading the model, deserializing the input data so that it could be passed to the model, and for getting predictions from the function. The code is implemented in test.py script.
The prediction was generated for one of the custom images using SageMaker runtime API with the invoke_endpoint method.

The screenshot below shows the active endpoint for the model.

![endpoints](C:\Users\ikulk\OneDrive - ETEC Solutions\Documents\z_UDACITY\AWS_ML_nanodegree\nd009t-c3-deep-learning-topics-within-computer-vision-nlp-project-starter\endpoints.JPG)

