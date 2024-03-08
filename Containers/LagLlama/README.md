# lagllama_demo
This demo includes a container for runing Lag Llama in Snowflake Container Services. The container includes the Lag Llama model and the GluonTS packages necessary for getting predictions. A sample notebook is included, but you will have to connect to your own time series data sources (a sample dataset is provided in the repo).

1. Build the Docker image
   `docker build --rm --platform linux/amd64 -t lagllama2 .`
                             
3. Tag and push it to Snowpark container services image repo
   docker tag lagllama2
   Based on the location of YOUR image folder:
   sfsenorthamerica-polaris2.registry.snowflakecomputing.com/mistral_vllm_db/public/images/lagllama2
   `docker push sfsenorthamerica-polaris2.registry.snowflakecomputing.com/mistral_vllm_db/public/images/lagllama2`
   
3. Push the lagllama_spec.yaml to YOUR stage. Make sure the spec file is set to your environment.

4. Create the service and navigate to the endpoint to access the jupyter environment.
   
## Time Series Modeling

1. Start in jupyter server
2. Navigate one level down to ll_github folder
3. Open the LagLlama.ipynb folder
4. To run the demo model, upload the data, timedata.csv which is in the github here
5. The notebook will show you the data, run predictions, plot the predictions, and provide evaluation statistics 



