# snowflake-notebooks
Unofficial snowflake notebooks created by me

This is an unofficial repo. If you need the latest and greatest, check out the snowflake [documentation](https://docs.snowflake.com/) or [Snowflake Developers](https://developers.snowflake.com/) or [Snowflake labs repo](https://github.com/Snowflake-Labs/).

## Notebooks

[Basics of Snowpark using Diabetes Dataset](SnowPark_Basics_Diabetes/Snowpark_For_Python_ML_Diabetes.ipynb)

[Basics of Snowpark with Forecasting Chicago Bus Ridership](Forecasting_ChicagoBus/Snowpark_Forecasting_Bus.ipynb)
  
---
### Scaling:

- [Big Datasets for showing off Snowpark](BigData_Demo/xgboost_tpcds.ipynb) 

- [Snowpark with Big Data for Forecasting](BigData_Demo/Snowpark_pandas_demo.ipynb)

- [Forecasting M5 Multi Time Series with Nixlta](TimeSeries/Forecasting_M5.ipynb)

  - [Streamlit Forecasting App](TimeSeries/TS_byseries_streamlit.ipynb)

--- 
### Cortex LLM  
- [Basics of Cortex LLM from Python](Cortex_LLM/Cortex_LLM_Python.ipynb)  

- [Evaluation of Cortex LLM with LangChain](Cortex_LLM/Cortex_LangChain.ipynb)

- [Streamlit App for Cortex LLM](Cortex_LLM/Streamlit_Cortex_LLM.ipynb)  

- [Simple RAG Example using Cortex functions with Streamlit App](RAG_Example/End-to-end_RAG_Snowflake.ipynb)

- [Evaluating RAG using Cortex by Generating Question / Answer Synthetic Pairs and a Model as Judge](Cortex_LLM/Cortex_RAG_Eval.ipynb)

- [Getting User Feedback using Cortex with Streamlit App](Cortex_LLM/Streamlit_Cortex_Evaluation_LLM.ipynb)

--- 
### ML Functions from Snowflake

- [Using Snowflake ML Functions for Forecasting](ML_Functions/Forecasting.ipynb)

---
### Feature Store:

- [Basic Feature Store in Snowflake](FeatureStore/Basic_Feature_Demo.ipynb)

- [Basics of Snowpark with Forecasting Chicago Bus Ridership including Feature Store](Forecasting_ChicagoBus/Snowpark_Forecasting_Bus_FeatureStore.ipynb)

---
### ML Tools:

- [OpenFE Tutorial / Demo](ML_Tools/Kaggle_OpenFE.ipynb)

---


### Misc Notebooks: 

- [Running Hugging Face Models off Snowpark](Snowpark_HuggingFace.ipynb)

- [Using Embedding Models or Sentence Transformers with Snowpark](sentence_transformers.ipynb)

- [Starter guide on UDFs and UDTFs in Snowflake](UDF_UDTF_Examples.ipynb)

- [Benchmarking VLLM with Mistral in Container Services](VLLM_benchmark_Mistral.ipynb)

- [Model Evaluation and Monitoring with Evidently AI in Snowflake](SnowPark_Basics_Diabetes/Diabetes_Evidently.ipynb)

- [ML Monitoring with Snowflake - Summit 2024 Talk](Summit2024_monitoring.ipynb)

- [Training examples using Snowpark ML and Stored Procedures](TrainingML.ipynb)

---
### Containers for Snowflake Container Services:

- [Mistral running on VLLM](Containers/Mistral_VLLM) and [blog post](https://medium.com/snowflake/generating-product-descriptions-with-mistral-7b-instruct-v0-2-with-vllm-serving-3fe7110b048b)

- [Lag LLama Time Series Model](Containers/LagLlama) and [github repo](https://github.com/rajshah4/Lagllama_demo)

- [H2O LLM Studio](Containers/H2O_LLM_Studio)

- Fine Tune Llama using Hugging Face TRL - [Container Services Setup Notebook](Containers/Container_setup_finetuneTRL.ipynb), [dockerfile](Containers/FineTuneTRL), and [Training Notebook using TRL](Containers/FineTuneTRL/LLamaTrain.ipynb)

- Use Nemo GuardRails with Snowflake Cortex - [based on this example](https://github.com/Snowflake-Labs/sf-samples/tree/cortex-nemoguardrails/samples/cortex-nemoguardrails), see [folder](Containers/NemoGuard)

---

## Using the Notebooks:  

For connecting to snowflake, the notebooks use a creds.json file.  You will want to create this JSON file with the following structure. My newer notebook also include code for using a private key (I typically use private key and 2 factor). 

```
{
    "account":"MY SNOWFLAKE ACCOUNT",
    "user": "MY USER",
    "password":"MY PASSWORD",
    "warehouse":"MY WH",
}
```