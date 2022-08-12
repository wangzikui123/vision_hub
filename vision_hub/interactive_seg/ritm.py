from pydoc import cli
import sys
import os
import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
import cv2
from copy import deepcopy
from IPython import embed
import re
from functools import partial

ritm_dir = os.path.join(os.path.dirname(__file__), 'ritm_interactive_segmentation')

sys.path.insert(0, ritm_dir)

from isegm.inference.predictors import get_predictor
from isegm.inference.clicker import Clicker
from isegm.utils import vis, exp
from isegm.inference import utils as ritm_utils
from isegm.inference.evaluation import evaluate_sample, infer_sample_no_gt
from isegm.inference.clicker import Click

class ritm_segmenter:
    def __init__(self, ritm_dir=ritm_dir, PROB_THRESH=0.49):
        device = torch.device('cpu')

        self.PROB_THRESH = PROB_THRESH

        model_dir = os.path.join(ritm_dir, "../../../../../weights")

        checkpoint_path = ritm_utils.find_checkpoint(model_dir, 'coco_lvis_h18s_itermask')
        model = ritm_utils.load_is_model(checkpoint_path, device)

        # Possible choices: 'NoBRS', 'f-BRS-A', 'f-BRS-B', 'f-BRS-C', 'RGB-BRS', 'DistMap-BRS'
        brs_mode = 'f-BRS-B'
        self.predictor = get_predictor(model, brs_mode, device, prob_thresh=self.PROB_THRESH)

        self.annotating_image = None
        # self.predictor.set_input_image(annotating_image)
        self.clicks_list = []
        self.pred_mask = None
    
    def auto_eval(self, img, mask, max_clicks=20, iou_thresh=0.85):
        clicks_list, ious_arr, pred = evaluate_sample(img, mask, self.predictor, 
                                              pred_thr=self.PROB_THRESH, 
                                              max_iou_thr=iou_thresh, max_clicks=max_clicks,
                                              init_clicks=None)
        assert len(clicks_list) == len(ious_arr) - 1 # initial IoU
        num_clicks = len(clicks_list)
        pred_mask = pred > self.PROB_THRESH
        
        return pred_mask, num_clicks

    def get_prediction_from_click(self, is_positive, coords):
        click = Click(is_positive=is_positive, coords=coords)
        self.clicks_list.append(click)
        clicker = Clicker(init_clicks=[click])
        with torch.no_grad():
            pred_probs = self.predictor.get_prediction(clicker)
            pred_mask = pred_probs > self.PROB_THRESH
        self.pred_mask = pred_mask
        draw = vis.draw_with_blend_and_clicks(self.annotating_image, mask=pred_mask, clicks_list=self.clicks_list)
        return draw

