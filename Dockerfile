FROM nvidia/cuda:11.4.0-cudnn8-devel-ubuntu18.04
CMD nvidia-smi

FROM cytomine/software-python3-base:v2.2.0

#INSTALL
RUN pip install numpy
RUN pip install shapely
RUN pip install torch
RUN pip install torchvision
RUN pip install pandas
RUN pip install matplotlib
RUN pip install pillow
RUN pip install tqdm
RUN pip install lmdb
RUN pip install albumentations
RUN pip install geopandas

RUN mkdir -p /models 
ADD /models/linear_model.pth /models/linear_model.pth
RUN chmod 444 /models/linear_model.pth

ADD ozanciga_tenpercent_resnet18.ckpt /models/ozanciga_tenpercent_resnet18.ckpt
RUN chmod 444 /models/ozanciga_tenpercent_resnet18.ckpt

COPY ["/simclr", "/app/simclr"]

#ADD FILES
RUN mkdir -p /app
ADD descriptor.json /app/descriptor.json
ADD run.py /app/run.py


ENTRYPOINT ["python3", "/app/run.py"]
