from ..poolers import ROIPooler
from ..backbone.resnet import BottleneckBlock, ResNet #同级目录引入
from .fast_rcnn import FastRCNNOutputLayers #同目录下引入
from .mask_head import build_mask_head

import shortuuid


@ROI_HEADS_REGISTRY.register()
class Res5ROIHeads(ROIHeads):
    def __init__(self, cfg, input_shape):
        super().__init__(cfg)

        self.in_features = cfg.MODEL.ROI_HEADS.IN_FEATURES
        pooler_resolution = cfg.MODEL.ROI_BOX_HEAD.POOLER_RESOLUTION
        pooler_type = cfg.MODEL.ROI_BOX_HEAD.POOLER_TYPE
        pooler_scales = (1.0 / input_shape[self.in_features[0]].stride,)
        sampling_ratio = cfg.MODEL.ROI_BOX_HEAD.POOLER_SAMPLING_RATIO
        self.mask_on = cfg.MODEL.MASK_ON
        self.enable_clustering = cfg.OWOD.ENABLE_CLUSTERING
        self.compute_energy_flag = cfg.OWOD.COMPUTE_ENERGY
        self.energy_save_path = os.path.join(cfg.OUTPUT_DIR, cfg.OWOD.ENERGY_SAVE_PATH)

        # Region of interest feature map pooler
        self.pooler = ROIPooler(
            output_size=pooler_resolution,
            scales=pooler_scales,
            sampling_ratio=sampling_ratio,
            pooler_type=pooler_type,
        )

        self.res5, out_channels = self._build_res5_block(cfg)
        # 预测头
        self.box_predictor = FastRCNNOutputLayers(
            cfg, ShapeSpec(channels=out_channels, height=1, width=1)
        )
        # 另一种头，没用
        if self.mask_on:
            self.mask_head = build_mask_dead(
                cfg,
                ShapeSpec(channels=out_channels, height=pooler_resolution, width=pooler_resolution)
            )

    # 主干网络
    def _build_res5_block(self, cfg):
        stage_channel_factor = 2 ** 3  # res5 is 8x res2
        num_groups = cfg.MODEL.RESNETS.NUM_GROUPS
        width_per_group = cfg.MODEL.RESNETS.WIDTH_PER_GROUP
        bottleneck_channels = num_groups * width_per_group * stage_channel_factor
        out_channels = cfg.MODEL.RESNETS.RES2_OUT_CHANNELS * stage_channel_factor
        stride_in_1x1 = cfg.MODEL.RESNETS.STRIDE_IN_1X1
        norm = cfg.MODEL.RESNETS.NORM

        blocks = ResNet.make_stage(
            BottleneckBlock,
            3,
            stride_per_block=[2, 1, 1],
            in_channels=out_channels // 2,
            bottleneck_channels=bottleneck_channels,
            out_channels=out_channels,
            num_groups=num_groups,
            norm=norm,
            stride_in_1x1=stride_in_1x1,
        )
        return nn.Sequential(*blocks), out_channels

    def _shared_roi_transform(self, features, boxes):
        x = self.pooler(features, boxes)
        return self.res5(x)

    # 计算energy
    def compute_energy(self, predictions, proposals):
        gt_classes = torch.cat([p.gt_classes for p in proposals])
        logits = predictions[0]
        data = (logits, gt_classes)
        location = os.path.join(self.energy_save_path, shortuuid.uuid() + '.pkl')
        torch.save(data, location)


    def forward(self, images, festures, proposals, targets=None):

        del images

        if self.trainging:
            proposals = self.label_and_sample_proposal(proposals, targets)
        del targets

        proposal_boxes = [x.proposal_boxes for x in proposals]
        box_features = self._shared_roi_tranform(
                [festures[f] for f in self.in_features], proposals
        )
        input_features = box_features.mean(dim=[2,3])
        predictions = self.box_predictor(input_features)

        if self.training:
            if self.enable_clustering:
                self.box_predictor.update_feature_store(input_features, proposals)
            del  festures
            if self.compute_energy_flag:
                self.compute_energy(predictions, proposals)
            losses = self.box_predictor.losses(predictions, proposals, input_features)
            if self.mask_on:
                pass
            return [], losses
        else:
            pred_instances, _ = self.box_predictor.inference(predictions, proposals)
            pred_instances = self.forward_with_given_boxed(festures, pred_instances)
            return pred_instances, {}

    # 预测
    def forward_with_given_boxes(self, features, instances):
        pass







