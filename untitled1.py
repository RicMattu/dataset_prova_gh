# -*- coding: utf-8 -*-
"""
Created on Mon Feb  3 16:36:47 2025

@author: User
"""
import json

file_json = "C00020598.json"  # Sostituisci con il tuo file JSON

with open(file_json, 'r') as f:
    json.load(f)
    
class MyDetection(datasets.CocoDetection):

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.edge = 480

        self.T = T.Compose([
            T.ToTensor(),
            T.Normalize(mean=[.485, .456, .406],
                        std=[.229, .224, .225]),
            T.Resize((self.edge, self.edge), antialias=True)
        ])

        self.T_target = preprocess_target

    def __getitem__(self, idx):
        img, target = super().__getitem__(idx)
        # PIL image
        w, h = img.size

        input_ = self.T(img)
        classes, boxes = self.T_target(target, w, h)

        return input_, (classes, boxes)