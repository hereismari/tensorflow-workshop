## How to run distributed tensorflow with Estimators + Experiment

Here are structions of how to run distributed tensorflow with data parallelism
for free (**no changes in the code to get it!!!**) using estimators and experiments with **TensorFlow v1.2 or higher**.

* In the local folder you'll find instructions about how to play with the
TF_CONFIG file locally running multiple python processes in the same machine

* In the distributed folder you'll find instructions about how to run
distributed tensorflow on basically any cloud environment including a full
tutorial of how to do it on Google Cloud.

For more about distributed tensorflow, go to:

* TensorFlow official doc:
  * [distributed tensorflow](https://www.tensorflow.org/deploy/distributed)
  * [estimators](https://www.tensorflow.org/extend/estimators)
  * [high-performance models](https://www.tensorflow.org/performance/performance_models)
* Talks:
  * [TensroFlow Ecosystem (TensorFlow Dev Summit 2017)](https://www.youtube.com/watch?v=yALzr4A2AzY)
  * [Distributed TensorFlow (TensorFlow Dev Summit 2017)](https://www.youtube.com/watch?v=la_M6bCV91M)
* Google Cloud ML Engine
  * [deployment](https://cloud.google.com/ml-engine/docs/how-tos/training-jobs)
  * [how to package your code](https://cloud.google.com/ml-engine/docs/how-tos/packaging-trainer)
  * [using GPUs](https://cloud.google.com/ml-engine/docs/how-tos/using-gpus)
  * [full example](https://github.com/GoogleCloudPlatform/cloudml-samples/tree/master/census)

* [Running Distributed TensorFlow in Kubernetes](https://github.com/amygdala/tensorflow-workshop/tree/1b9a868201c5d0d19b54e320cb2560e08340a916/workshop_sections/distributed_tensorflow)
* [TensorFlow Ecosystem - This repository contains examples for integrating TensorFlow with other open-source frameworks.](https://github.com/tensorflow/ecosystem)

*Special thanks to: @elibixby and @xiejw*
