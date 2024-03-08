# snowflake-notebooks
Unofficial snowflake notebooks created by me

This is an unofficial repo. If you need the latest and greatest, check out the snowflake [documentation](https://docs.snowflake.com/) or [Snowflake Developers](https://developers.snowflake.com/) or [Snowflake labs repo](https://github.com/Snowflake-Labs/).

## Notebooks

[Basics of Snowpark using Diabetes Dataset](SnowPark_Basics_Diabetes/Snowpark_For_Python_ML_Diabetes.ipynb)

[Basics of Snowpark with Forecasting Chicago Bus Ridership](Forecasting_ChicagoBus/Snowpark_Forecasting_Bus.ipynb)
  
---
### Scaling:

- [Big Datasets for showing off Snowpark](BigData_Demo/xgboost_tpcds.ipynb) 

- [Forecasting M5 Multi Time Series with Nixlta](TimeSeries_M5/Forecasting_M5.ipynb)

  - [Streamlit Forecasting App](TimeSeries_M5/TS_byseries_streamlit.ipynb)

--- 
### Cortex LLM  
- [Basics of Cortex LLM from Python](Cortex_LLM/Cortex_LLM_Python.ipynb)  

- [Evaluation of Cortex LLM with LangChain](Cortex_LLM/Cortex_LangChain.ipynb)

- [Streamlit App for Cortex LLM](Cortex_LLM/Streamlit_Cortex_LLM.ipynb)  

- [Simple RAG Example using Cortex functions with Streamlit App](RAG_Example/End-to-end_RAG_Snowflake.ipynb)


---
### Misc Notebooks:

- [Running Hugging Face Models off Snowpark](Snowpark_HuggingFace.ipynb)

- [Starter guide on UDFs and UDTFs in Snowflake](UDF_UDTF_Examples.ipynb)

- [Benchmarking VLLM with Mistral in Container Services](VLLM_benchmark_Mistral.ipynb)

- [Model Evaluation and Monitoring with Evidently AI in Snowflake](SnowPark_Basics_Diabetes/Diabetes_Evidently.ipynb)

---
### Containers for Snowflake Container Services:

- [Mistral running on VLLM](Containers/Mistral_VLLM) and [blog post](https://medium.com/snowflake/generating-product-descriptions-with-mistral-7b-instruct-v0-2-with-vllm-serving-3fe7110b048b)

- [Lag LLama Time Series Model](Containers/LagLlama) and [github repo](https://github.com/rajshah4/Lagllama_demo)

- [H2O LLM Studio](Containers/H2O_LLM_Studio)

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