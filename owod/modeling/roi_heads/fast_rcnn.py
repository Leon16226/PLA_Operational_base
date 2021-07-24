import torch.nn as nn


class FastRCNNOutputs:

    def __init__(self,
                 box2box_transform,
                 pred_class_logits,  # 存储逻辑回归值
                 pred_proposal_deltas,
                 proposals,
                 invalid_class_range,
                 smooth_l1_beta=0.0,
                 box_reg_loss_type="smooth_l1",
                 ) -> None:
        super().__init__()
        self.box2box_transform = box2box_transform
        self.num_preds_per_image = [len(p) for p in proposals]
        self.pred_class_logits = pred_class_logits
        self.pred_proposal_deltas = pred_proposal_deltas
        self.smooth_l1_beta = smooth_l1_beta
        self.box_reg_loss_type = box_reg_loss_type

        self.image_shapes = [x.image_size for x in proposals]
        self.invalid_class_range = invalid_class_range

        if len(proposals):
            box_type = type(proposals[0].proposal_boxes)
            self.proposals = box_type.cat([p.proposal_boxes for p in proposals])

        # The following fields should exist only when training.
        if proposals[0].has("gt_boxes"):
            self.gt_boxes = box_type.cat([p.gt_boxes for p in proposals])
            assert proposals[0].has("gt_classes")
            self.gt_classes = cat([p.gt_classes for p in proposals], dim=0)
        else:
            self.proposals = Boxes(torch.zeros(0, 4, device=self.pred_proposal_deltas.device))
        self._no_instances = len(proposals) == 0  # no instances found

    def losses(self):
        """
        Compute the default losses for box head in Fast(er) R-CNN,
        with softmax cross entropy loss and smooth L1 loss.

        Returns:
            A dict of losses (scalar tensors) containing keys "loss_cls" and "loss_box_reg".
        """
        return {"loss_cls": self.softmax_cross_entropy_loss(), "loss_box_reg": self.box_reg_loss()}

    def _log_accuracy(self):
        num_instances = self.gt_classes.numel()
        pred_classes = self.pred_class_logits.argmax(dim=1)
        bg_class_ind = self.pred_class_logits.shape[1] - 1

        fg_inds = (self.gt_classes >= 0) & (self.gt_classes < bg_class_ind)
        num_fg = fg_inds.nonzero().numel()
        fg_gt_classes = self.gt_classes[fg_inds]
        fg_pred_classes = pred_classes[fg_inds]






    def softmax_cross_entropy_loss(self):
        if self._no_instances:
            return 0.0 * self.pred_class_logits.sum()
        else:
            self._log_accuracy()




class FastRCNNOutputLayers(nn.Module):

    @configurable
    def __init__(
            self,
            input_shape: ShapeSpec,
            *,
            box2box_transofrm,
            clustering_items_per_class,
            clustering_start_iter,
            clustering_update_mu_iter,
            clustering_momentum,
            clustering_z_dimension,
            enable_clustering,
            prev_intro_cls,
            curr_intro_cls,
            max_iterations,
            output_dir,
            feat_store_path,
            margin,
            num_classes: int,
            test_score_thresh: float = 0.0,
            test_nms_thresh: float = 0.5,
            test_topk_per_image: int = 100,
            cls_agnostic_bboc_reg: bool = False,
            smooth_l1_beta: float = 0.0,
            box_reg_loss_type: str = "smooth_l1",
            loss_weight: Union[float, Dict[str, float]] = 1.0,
    ):
        super().__init__()
        if isinstance(input_shape, int):
            input_shape = ShapeSpec(channels=input_shape)
        input_size = input_shape.channels * (input_shape.width or 1) * (input_shape.height or 1)
        #
        self.cls_score = Linear(input_size, num_classes + 1)  # 21
        num_bbox_reg_classes = 1 if cls_agnostic_bbox_reg else num_classes
        box_dim = len(box2box_transform.weights)  # 21 * 4 = 84
        self.bbox_pred = Linear(input_size, num_bbox_reg_classes * box_dim)

        # 初始化
        self.box2box_transform = box2box_transform
        self.smooth_l1_beta = smooth_l1_beta
        self.test_score_thresh = test_score_thresh
        self.test_nms_thresh = test_nms_thresh
        self.test_topk_per_image = test_topk_per_image
        self.box_reg_loss_type = box_reg_loss_type
        if isinstance(loss_weight, float):
            loss_weight = {"loss_cls": loss_weight, "loss_box_reg": loss_weight}
        self.loss_weight = loss_weight

        self.num_classes = num_classes
        self.clustering_start_iter = clustering_start_iter
        self.clustering_update_mu_iter = clustering_update_mu_iter
        self.clustering_momentum = clustering_momentum

        self.hingeloss = nn.HingeEmbeddingLoss(2)
        self.enable_clustering = enable_clustering

        self.prev_intro_cls = prev_intro_cls
        self.curr_intro_cls = curr_intro_cls
        self.seen_classes = self.prev_intro_cls + self.curr_intro_cls
        self.invalid_class_range = list(range(self.seen_classes, self.num_classes - 1))

        self.max_iterations = max_iterations
        self.feature_store_is_stored = False
        self.output_dir = output_dir
        self.feat_store_path = feat_store_path
        self.feature_store_save_loc = os.path.join(self.output_dir, self.feat_store_path, 'feat.pt')

        # 创造F_store文件，开始逐渐evolve
        if os.path.isfile(self.feature_store_save_loc):
            # 从文件中加载 F_store list
            self.feature_store = torch.load(self.feature_store_save_loc)
        else:
            # 初始化 a feature store F_store list
            self.feature_store = Store(num_classes + 1, clustering_items_per_class)
        self.means = [None for _ in range(num_classes + 1)]
        self.margin = margin

    def forward(self, x):
        if x.dim() > 2:
            x = torch.flatten(x, start_dim=1)
        scores = self.cls_score(x)
        proposal_deltas = self.bbox_pred(x)
        return scores, proposal_deltas

    # 跟新 feature_store
    def updata_feature_store(self, features, proposals):
        gt_classes = torch.cat([p.get_classes for p in proposals])
        # P : the set of class prototypes
        # F_store : 在队列中存储相应类别的特征向量
        # F_store -> 取平均值 -> P

        # feature: the class specific features
        # gt_classes: 类别对应的id
        self.feature_store.add(features, gt_classes)

        storage = get_event_storage()

        # 存储
        torch.save(self.feature_store, self.feature_store_save_loc)
        self.feature_store_is_stored = True

    def clstr_loss_l2_cdist(self, input_features, proposals):
        gt_classes = torch.cat([p.gt_classes for p in proposals])
        # num_classes ：前景类别
        mask = gt_classes != self.num_classes
        fg_features = input_features[mask]
        classes = gt_classes[mask]

        all_means = self.means
        for item in all_means:
            if item != None:
                length = item.shape
                break

        for i, item in enumerate(all_means):
            if item == None:
                all_means[i] = torch.zeros((length))

        # D(f_c,p_i) : l1 diatance function  任何距离函数
        distances = torch.cdist(fg_features, torch.stack(all_means).cuda(), p=self.margin)
        labels = []

        for index, feature in enumerate(fg_features):
            for cls_index, mu in enumerate(self.means):
                if mu is not None and feature is not None:
                    if classes[index] == cls_index:
                        labels.append(1)
                    else:
                        labels.append(-1)
                else:
                    labels.append(0)

        hingeloss = nn.HingeEmbeddingLoss(2)
        loss = self.hingeloss(distances, torch.tensor(labels).reshape((-1, self.num_classes + 1)).cuda())

        return loss

    def get_clustering_loss(self, input_features, proposals):
        storage = get_event_storage()
        c_loss = 0
        # I_b之后才开始计算clustering loss
        if storage.iter == self.clustering_start_iter:
            items = self.feature_store.retrieve(-1)
            for index, item in enumerate(items):
                if len(item) == 0:
                    self.means[index] = None
                else:
                    mu = torch.tensor(item).mean(dim=0)
                    self.means[index] = mu
            # loss func
            c_loss = self.clstr_loss_l2_cdist(input_features, proposals)

    def losses(self, predictions, proposals, input_features=None):

        if input_features is not None:
            losses["loss_clustering"] = self.get_clustering_loss(input_features, proposals)
