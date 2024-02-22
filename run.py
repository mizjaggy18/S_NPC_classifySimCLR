from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals

import logging
import sys
from argparse import ArgumentParser
import logging
import shutil

import os
import numpy as np
from shapely.geometry import shape, box, Polygon, Point, MultiPolygon, LineString
from shapely import wkt
from shapely.ops import split
import geopandas

from glob import glob
from tifffile import imread

import cytomine
from cytomine import Cytomine, CytomineJob
from cytomine.models import Property, Annotation, AnnotationTerm, AnnotationCollection, Job, JobData, TermCollection, ImageInstanceCollection
import torch
# from torchvision.models import DenseNet
def load_model_weights(model, weights):

    model_dict = model.state_dict()
    weights = {k: v for k, v in weights.items() if k in model_dict}
    if weights == {}:
        print('No weight could be loaded..')
    model_dict.update(weights)
    model.load_state_dict(model_dict)

    return model

from simclr.model18 import Model, Identity
from simclr.utils import distribute_over_GPUs, validate_arguments
import torch.nn as nn
import random

# from augmentations import get_transforms, torchvision_transforms, album_transforms, AlbumentationsTransform, get_rgb_transforms

torch.backends.cudnn.benchmark=True

torch.no_grad()
seed = 44 #44 2022
random.seed(seed)
np.random.seed(seed)
torch.manual_seed(seed)
torch.cuda.manual_seed(seed)
torch.cuda.manual_seed_all(seed)

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print(device)

class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        # model_path="G:/My Drive/MMU Postdoc/ISDB Project/Codes and algorithms/Python Codes/Cytomine-python-client/examples/Classification-related/PyTorch/ozanciga_tenpercent_resnet18.ckpt" #ozanciga-pretrained
        model_path="./models/ozanciga_tenpercent_resnet18.ckpt"

        # Load pre-trained model
        base_model = Model(pretrained=False)
        if (not False) and (not False):
            print('Loading base model from ', model_path)

            # load model trained from ozanciga       
            state = torch.load(model_path, map_location='cuda:0')

            state_dict = state['state_dict']
            # print(state_dict)
            for key in list(state_dict.keys()):
                state_dict[key.replace('model.', '').replace('resnet.', 'f.')] = state_dict.pop(key)            # 
            base_model = load_model_weights(base_model, state_dict)

        self.f = base_model.f
        NUM_CLASSES=4
        self.fc = nn.Linear(512, NUM_CLASSES, bias=True)
        

    def forward(self, x):
        x = self.f(x)
        feature = torch.flatten(x, start_dim=1)
        out = self.fc(feature)
        return out



from PIL import Image
import matplotlib.pyplot as plt
import time
import cv2
import math

from shapely import affinity
from shapely.geometry.multipolygon import MultiPolygon
from scipy.spatial import Voronoi



__author__ = "WSH Munirah W Ahmad <wshmunirah@gmail.com>"
__version__ = "2.1.0"
# Date created: 20 Oct 2022


def run(cyto_job, parameters):
    logging.info("Entering run(cyto_job=%s, parameters=%s)", cyto_job, parameters)

    job = cyto_job.job
    user = job.userJob
    project = cyto_job.project
    NUM_CLASSES=4

    job.update(status=Job.RUNNING, progress=10, statusComment="Initialization...")

    
    # ---- load linear model ---
    # linear_path="G:/My Drive/MMU Postdoc/Fareeds Work/Results and Codes/dataset/NPC/NPC-09032023/4-ozanciga-linear_100ep/linear_model.pth" #model-4
    model = Net()
    linear_path="./models/linear_model.pth"
    print("Loading linear (finetuned) model from: ",linear_path)
    
    state_dict = torch.load(linear_path, map_location='cuda:0')
    for key in list(state_dict.keys()):
        state_dict[key.replace('module.', '').replace('resnet.', 'f.')] = state_dict.pop(key)
    model = load_model_weights(model, state_dict)

    model.cuda()
    model.eval()
    print(f"total params: \t{sum([np.prod(p.size()) for p in model.parameters()])}")
    #---------------------------


    terms = TermCollection().fetch_with_filter("project", parameters.cytomine_id_project)    
    print(terms)
    for term in terms:
        print("ID: {} | Name: {}".format(
            term.id,
            term.name
        )) 
    job.update(status=Job.RUNNING, progress=20, statusComment="Terms collected...")
    

    images = ImageInstanceCollection().fetch_with_filter("project", project.id)
    
    list_imgs = []
    if parameters.cytomine_id_images == 'all':
        for image in images:
            list_imgs.append(int(image.id))
    else:
        list_imgs = [int(id_img) for id_img in parameters.cytomine_id_images.split(',')]
        print('Images: ', list_imgs)
    job.update(status=Job.RUNNING, progress=30, statusComment="Images gathered...")
             
    id_project = parameters.cytomine_id_project
    id_term = parameters.cytomine_id_roi_term
    
    working_path = os.path.join("tmp", str(job.id))
    
    if not os.path.exists(working_path):
        logging.info("Creating working directory: %s", working_path)
        os.makedirs(working_path)
    try:
        for id_image in list_imgs:
            print('Parameters (id_project, id_image, id_term, id_term_poly):',id_project, id_image, id_term)

            roi_annotations = AnnotationCollection()
            roi_annotations.project = id_project
            roi_annotations.image = id_image
            roi_annotations.term = id_term
            roi_annotations.showWKT = True
            roi_annotations.showMeta = True
            roi_annotations.showGIS = True
            roi_annotations.showTerm = True
            roi_annotations.fetch()
            print(roi_annotations)

            pred_c0 = 0
            pred_c1 = 0
            pred_c2 = 0
            pred_c3 = 0
            id_terms = 0

            job.update(status=Job.RUNNING, progress=40, statusComment="Running splitpoly on ROI-WSI...")

            for i, roi in enumerate(roi_annotations):
                #Get Cytomine ROI coordinates for remapping to whole-slide
                #Cytomine cartesian coordinate system, (0,0) is bottom left corner                
                print("----------------------------Classify SimCLR------------------------------")
                roi_geometry = wkt.loads(roi.location)
                # print("ROI Geometry from Shapely: {}".format(roi_geometry))
                print("ROI Bounds")
                print(roi_geometry.bounds)

                #Dump ROI image into local PNG file
                # roi_path=os.path.join(working_path,str(roi_annotations.project)+'/'+str(roi_annotations.image)+'/'+str(roi.id))
                roi_path=os.path.join(working_path,str(roi_annotations.project)+'/'+str(roi_annotations.image)+'/')
                print(roi_path)
                roi_png_filename=os.path.join(roi_path+str(roi.id)+'.png')
                print("roi_png_filename: %s" %roi_png_filename)
                roi.dump(dest_pattern=roi_png_filename)
                # im=Image.open(roi_png_filename)

                # check white patches
                J = cv2.imread(roi_png_filename,0) #read image and convert to grayscale    
                [r, c]=J.shape

                if r > 256 or c > 256:
                    JC = cv2.imread(roi_png_filename) #read image in RGB
                    scale_percent = .5 #.5 is 50%
                    width = int(c * scale_percent)
                    height = int(r * scale_percent)
                    dim = (width, height)                        
                    JC2 = cv2.resize(JC, dim, interpolation = cv2.INTER_AREA)

                else:
                    JC2 = cv2.imread(roi_png_filename) #read image in RGB
                
                #Start classification
                im = cv2.cvtColor(JC2,cv2.COLOR_BGR2RGB)
                im = cv2.resize(im,(224,224))
                im = im.reshape(-1,224,224,3)
                output = np.zeros((0,NUM_CLASSES))
                arr_out_gpu = torch.from_numpy(im.transpose(0, 3, 1, 2)).type('torch.FloatTensor').to(device)
                output_batch = model(arr_out_gpu)
                output_batch = output_batch.detach().cpu().numpy()                
                output = np.append(output,output_batch,axis=0)
                pred_labels = np.argmax(output, axis=1)
                # pred_labels=[pred_labels]

                if pred_labels[0]==0:
                    # print("Class 0: Normal")
                    id_terms=parameters.normal_term
                    pred_c0=pred_c0+1
                    # roi.dump(dest_pattern=os.path.join(roi_path+'Class0/'+str(roi.id)+'.png'),alpha=True)
                elif pred_labels[0]==1:
                    # print("Class 1: LHP")
                    id_terms=parameters.lhp_term
                    pred_c1=pred_c1+1
                    # roi.dump(dest_pattern=os.path.join(roi_path+'Class1/'+str(roi.id)+'.png'),alpha=True)
                elif pred_labels[0]==2:
                    # print("Class 2: NPI")
                    id_terms=parameters.npi_term
                    pred_c2=pred_c2+1
                    # roi.dump(dest_pattern=os.path.join(roi_path+'Class2/'+str(roi.id)+'.png'),alpha=True)
                elif pred_labels[0]==3:
                    # print("Class 3: NPC")
                    id_terms=parameters.npc_term
                    pred_c3=pred_c3+1
            
                if id_terms!=0:
                    cytomine_annotations = AnnotationCollection()
                    annotation=roi_geometry                    
                    cytomine_annotations.append(Annotation(location=annotation.wkt,#location=roi_geometry,
                                                        id_image=id_image,#conn.parameters.cytomine_id_image,
                                                        id_project=project.id,
                                                        id_terms=[id_terms]))
                    print(".",end = '',flush=True)

                    #Send Annotation Collection (for this ROI) to Cytomine server in one http request
                    ca = cytomine_annotations.save()        
                                           
    finally:
        job.update(progress=100, statusComment="Run complete.")
        shutil.rmtree(working_path, ignore_errors=True)
        logging.debug("Leaving run()")
        
if __name__ == "__main__":
    logging.debug("Command: %s", sys.argv)

    with cytomine.CytomineJob.from_cli(sys.argv) as cyto_job:
        run(cyto_job, cyto_job.parameters)

                  






