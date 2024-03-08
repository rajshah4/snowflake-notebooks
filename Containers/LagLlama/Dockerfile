FROM nvcr.io/nvidia/pytorch:23.06-py3

# Set the working directory
WORKDIR /workspace/

# For Jupyter only - remove once validated
EXPOSE  8888

COPY --chmod=777 requirements.txt entrypoint.sh ./ 

# Install the dependencies from the requirements.txt file
RUN pip install --no-cache-dir --upgrade pip
RUN pip install --no-cache-dir -r requirements.txt
RUN pip install --upgrade huggingface_hub

RUN mkdir /workspace/.local /workspace/.cache && chmod 777 -R /workspace

#ts code
RUN apt-get install -y git
RUN git clone https://github.com/time-series-foundation-models/lag-llama ll_github
RUN git clone https://github.com/Nixtla/nixtla nitlxa

#model
RUN apt-get update && apt-get install -y wget
RUN wget https://huggingface.co/time-series-foundation-models/Lag-Llama/resolve/main/lag-llama.ckpt

COPY --chmod=777 LagLlama.ipynb ll_github/. 


ENTRYPOINT ["./entrypoint.sh"]
