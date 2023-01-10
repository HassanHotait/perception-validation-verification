from turtle import forward
import torch
from torch import nn

from SMOKE.smoke.structures.image_list import to_image_list

from ..backbone import build_backbone
from ..heads.heads import build_heads


class KeypointDetector(nn.Module):
    '''
    Generalized structure for keypoint based object detector.
    main parts:
    - backbone
    - heads
    '''

    def __init__(self, cfg):
        super(KeypointDetector, self).__init__()

        self.backbone = build_backbone(cfg)
        self.heads = build_heads(cfg, self.backbone.out_channels)

    def forward(self, images, targets=None,default=True,method="dataset_depth_ref",frame_id="All"):
        """
        Args:
            images:
            targets:

        Returns:

        """
        #print("targets in forwards",targets)
        print("Type of Targets in forward: ",type(targets))
        #print("Format of Targets in forward: ",format(targets))
        #print("Target hm field in forward",targets[0].get_field("hm"))

        if self.training and targets is None:
            raise ValueError("In training mode, targets should be passed")
        print("Type of Images in model: ",type(images))
        images = to_image_list(images)
        print("Type of images in forward",type(images))
        print("Tensor Size: ",images.tensors.size())
        features = self.backbone(images.tensors)

        #print("Features in forward: ",features)
        # print("Type of features in forward",type(features))
        # #print("Format of features in forward",format(features))
        # print("Length of Targets: ",len(targets))
        # # hm_key=targets[0].get_field("hm")
        # # print("HM Key Test: ",hm_key)
        
        result, detector_losses = self.heads(features, targets,default,method,frame_id) # Here is the issue, targets has no field hm for infer while for original code it works

        #print("Result: ",result)

        if self.training:
            losses = {}
            losses.update(detector_losses)

            return losses

        return result