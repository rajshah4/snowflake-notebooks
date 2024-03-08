FROM nvcr.io/nvidia/pytorch:23.06-py3

# Set the working directory
WORKDIR /workspace/

# For Jupyter only - remove once validated
EXPOSE  8888

COPY --chmod=777 requirements.txt entrypoint.sh Product_Descriptions.ipynb ./ 

# Install the dependencies from the requirements.txt file
RUN pip install --no-cache-dir --upgrade pip
RUN pip install --no-cache-dir -r requirements.txt
RUN pip install --upgrade huggingface_hub

RUN mkdir /workspace/.local /workspace/.cache && chmod 777 -R /workspace

ENTRYPOINT ["./entrypoint.sh"]
