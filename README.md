# snowflake-notebooks
Unofficial snowflake notebooks created by me

This is an unofficial repo. If you need the latest and greatest, check out the snowflake [documentation](https://docs.snowflake.com/) or their [official repos](https://github.com/snowflakedb) or [Snowflake labs repo](https://github.com/Snowflake-Labs/).

## Notebooks

[Basics of Snowpark using Diabetes Dataset](SnowPark_Basics_Diabetes/Snowpark_For_Python_ML_Diabetes.ipynb)  

[Big Datasets for showing off Snowpark](BigData_Demo/xgboost_tpcds.ipynb) 

[Running Hugging Face Models off Snowpark](Snowpark_HuggingFace.ipynb)

[Starter guide on UDFs and UDTFs in Snowflake](UDF_UDTF_Examples.ipynb)

[Benchmarking VLLM with Mistral in Container Services](VLLM_benchmark_Mistral.ipynb)

[Model Evaluation and Monitoring with Evidently AI in Snowflake](SnowPark_Basics_Diabetes/Diabetes_Evidently.ipynb)

[Simple RAG Example using Cortex functions](RAG_Example/End-to-end_RAG_Snowflake.ipynb)


## Using the Notebooks

For connecting to snowflake, the notebooks use a creds.json file.  You will want to create this JSON file with the following structure
```
{
    "account":"MY SNOWFLAKE ACCOUNT",
    "user": "MY USER",
    "password":"MY PASSWORD",
    "warehouse":"MY WH",
}
```

## Containers

[Lag-LLama in Snowflake Container Services](https://github.com/rajshah4/Lagllama_demo)