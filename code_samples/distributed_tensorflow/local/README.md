## Running a TensorFlow experiment locally

1. Build your estimator model and make sure to run it using an
   experiment like:
   
   *learn_runner.run(generate_experiment_fn(), run_config=tf.contrib.learn.RunConfig())*

	Notice that Tensorflow should be v1.2 or higher and 
	the model can be **any estimators model + experiment!!**

2. Set up the TF_CONFIG env. variable, in this variable you'll define
   how your achitecture is defined.
   
   Example:

   ```python
   cluster = {'master': ['localhost:2220'],
              'ps': ['localhost:2222', 'localhost:2221'],
	      'worker': ['localhost:2223', 'localhost:2224']}

   TF_CONFIG = json.dumps(
	  {'cluster': cluster,
	   'task': {'type': args.task_type, 'index': args.task_index},
	   'model_dir': '/tmp/output_test',
	   'environment': 'cloud'
	  })
   ```

   I'll explain briefly what each parameter defined above means:

   ### Cluster
   
   This is defined as a cluster spec (more about it [here](https://www.tensorflow.org/deploy/distributed)).
   In this cluster spec we have 1 master, 2 ps and 2 workers.
   
   Where:
   
	* ps: saves the parameters among all workers.
	All workers can read/write/update the parameters for model via ps
	as some models are extremely large the parameters are shared to
	store on ps (each stores a subset).
	
	* worker: does the training.
	
	* master: is basically a special worker, that does training, but
	is the only worker who also does ckpt restore/saving and evaluation.
   
   ### Task
   
   This define what is the role of the current node (or process) and
   that's why we're using args to define this, because it changes
   for each node. A task example would be: 
   
   *'task': {'type': 'worker', 'index': 0}*
   
   This says that the node that has this TF_CONFIG is the worker on
   index 0 defined in the cluster spec.
   
   ### Model_dir
   
   This is the path where the master should save the checkpoints and
   other model related files. Since we're running locally we can use
   a local path. But for a multi machines environment a distributed
   file system should be used, so we would have something like:
   
   * For Google Storage: *gs://<bucket>/<path>*
   * For DFS: *dfs://<path>*
   
   ### Environment
   
   By the default it's "local", but since we want to run distributed
   we'll use "cloud".
   
   
3. Once we have a TF_CONFIG variable defined correctly, we can run the
   experiment.  
   
   The way that this will work is:
   
   For each node (process) you'll need to run:
   
   *export TF_CONFIG=`python TF_CONFIG.py <task_type> <task_index>`*
   `python experiment.py`
   
4. An example of output for each process is available at [imgs/](https://github.com/mari-linhares/tensorflow-workshop/tree/master/code_samples/distributed_tensorflow/local/imgs)
