FROM pytorch/pytorch:1.8.1-cuda11.1-cudnn8-runtime
RUN apt-get update && apt-get install git sudo -y
RUN mkdir /hot
RUN git clone https://jw9730/hot.git /hot
WORKDIR /hot
RUN pip3 install torch==1.8.1+cu111 torchvision==0.9.1+cu111 torchaudio==0.8.1 -f https://download.pytorch.org/whl/torch_stable.html
RUN pip install -r requirements.txt
RUN pip install torch-sparse==0.6.9 torch-scatter==2.0.6 -f https://data.pyg.org/whl/torch-1.8.0+cu111.html
RUN pip install torch-geometric==1.6.3
CMD ["/bin/bash"]
