## Running a TensorFlow experiment on the cloud

1. Build your estimator model and make sure to run it using an
   experiment like:
   
   *learn_runner.run(generate_experiment_fn(), run_config=tf.contrib.learn.RunConfig())*

	Notice that Tensorflow should be v1.2 or higher and
	the model can be **any estimators model + experiment!!**

2. Set up the TF_CONFIG env. variable, in this variable you'll define
   how your achitecture is defined.
   
   Example:

   ```python
   cluster = {'master': ['instance-1:8000'],
              'ps': ['instance-2:8000'],
              'worker': ['instance-3:8000']}

   TF_CONFIG = json.dumps(
	  {'cluster': cluster,
	   'task': {'type': args.task_type, 'index': args.task_index},
	   'model_dir': 'gs://bucket/your/path',
	   'environment': 'cloud'
	  })
   ```

   I'll explain briefly what each parameter defined above means:

   ### Cluster
   
   This is defined as a cluster spec (more about it [here](https://www.tensorflow.org/deploy/distributed)).
   In this cluster spec we have 1 master, 1 ps and 1 worker.
   
   Where:
   
	* ps: saves the parameters among all workers.
	All workers can read/write/update the parameters for model via ps
	as some models are extremely large the parameters are shared to
	store on ps (each stores a subset).
	
	* worker: does the training.
	
	* master: is basically a special worker, that does training, but
	is the only worker who also does ckpt restore/saving and evaluation.
   
   ### Task
   
   This define what is the role of the current node and
   that's why we're using args to define this, because it changes
   for each node. A task example would be: 
   
   *'task': {'type': 'worker', 'index': 0}*
   
   This says that the node that has this TF_CONFIG is the worker on
   index 0 defined in the cluster spec.
   
   ### Model_dir
   
   This is the path where the master should save the checkpoints and
   other model related files. For a multi machines environment
   a distributed file system should be used, so we'll have something like:
   
   * For Google Storage: *gs://<bucket>/<path>*
   * For DFS: *dfs://<path>*
   
   ### Environment
   
   By the default it's "local", but since we want to run distributed
   we'll use "cloud".
 
3. Copy the experiment.py file and the TF_CONFIG.py file to each
   instance.

4. Enable communication between the instances. In Google cloud:
   
   1. Go to console  
   2. Compute > networking > firewall rules  
   3. Add an ingress rule to allow access into the ports
       defined on the TF_CONFIG file  

5. Create a directory for the model checkpoint files in a distributed
   storage, it can be:

   * Google cloud storage
      1. Create a bucket
      2. Create a directory inside this bucket
      3. Set model_dir in the TF_CONFIG to a path like: gs://<bucket>/<path_to_dir>/
      4. Make sure the master is allowed to write in the bucket
         1. Considering that you’ve created the bucket in the
            same project of your instance you need to stop the
            instance, change the instance scope to allow full
            access to all cloud APIs
            (Edit > allow full access to all cloud APIs) and restart it.
   * DFS
      1. You should make sure the master is allowed to write 
         in the DFS dir

3. Once we have a TF_CONFIG variable defined correctly, all VMs can
   talk to each other and the master has write access to the
   distributed file system, we can run the experiment.  
   
   The way that this will work is:
   
   For each node (instance) you'll need to run:
   
   `export TF_CONFIG=python TF_CONFIG.py <task_type> <task_index>`
   `python experiment.py`
   
4. The output for each node shoud be silimar to the ones at [../local/imgs/](https://github.com/mari-linhares/tensorflow-workshop/tree/master/code_samples/distributed_tensorflow/local/imgs)

## Google Cloud ML Engine

A simpler approach is to use Google Cloud ML Engine to run your jobs, check:

* [deployment](https://cloud.google.com/ml-engine/docs/how-tos/training-jobs)
* [how to package your code](https://cloud.google.com/ml-engine/docs/how-tos/packaging-trainer)
* [using GPUs](https://cloud.google.com/ml-engine/docs/how-tos/using-gpus)
* [full example](https://github.com/GoogleCloudPlatform/cloudml-samples/tree/master/census)

