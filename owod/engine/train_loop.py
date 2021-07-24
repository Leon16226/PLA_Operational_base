class TrainerBase:
    def __init__(self) -> None:
        super().__init__()
        self._hooks = []

    # OUTPUT_DIR: "./output/temp_3"
    # ENERGY_SAVE_PATH: 'energy'
    def analyse_energy(self, temp=1.5):
        files = os.listdir(os.path.join(self.cfg.OUTPUT_DIR, self.cfg.OWOD.ENERGY_SAVE_PATH))
        # TEMPERATURE: 1.5
        temp = self.cfg.OWOD.TEMPERATURE

        unk = []
        known = []

        for id, file in enumerate(files):
            path = os.path.join(self.cfg.OUTPUT_DIR, self.cfg.OWOD.ENERGY_SAVE_PATH, file)
            try:
                logits, classes = torch.load(path)
            except:
                logger.info('Not able to load ' + path + ". Continuing...")
                continue
            # Task1：
            # PREV_INTRODUCED_CLS: 0
            # CUR_INTRODUCED_CLS: 20

            # Task2:
            # PREV_INTRODUCED_CLS: 20
            # CUR_INTRODUCED_CLS: 20

            # Task3
            # PREV_INTRODUCED_CLS: 40
            # CUR_INTRODUCED_CLS: 20

            # 就是模型总的看见过的图片
            num_seen_classes = self.cfg.OWOD.PREV_INTRODUCTED_CLS + self.cfg.OWOD.CUR_INTRODUCED_CLS
            lse = temp * torch.logsumexp(logits[:, :num_seen_classes] / temp, dim=1)

            for i, cls in enumerate(classes):
                if cls == self.cfg.MODEL.ROI_HEADS.NUM_CLASSES:
                    continue
                if cls == self.cfg.MODEL.ROI_HEADS.NUM_CLASSES - 1:
                    unk.append(lse[i].detach().cpu().tolist())
                else:
                    known.append(lse[i].detach().cpu().tolist())

            wb_dist_param = []

            start_time = time.time()
            wb_unk = Fit_Weibull_3P(failures=unk, show_probability_plot=False, print_results=False)
            logger.info("--- %s seconds ---" % (time.time() - start_time))

            start_time = time.time()
            wb_known = Fit_Weibull_3P(failures=known, show_probability_plot=False, print_results=False)
            logger.info("--- %s seconds ---" % (time.time() - start_time))

            wb_dist_param.append(
                {"scale_known": wb_known.alpha, "shape_known": wb_known.beta, "shift_known": wb_known.gamma})

            # 存储
            torch.save(wb_dist_param, param_save_location)

            logger.info('Plotting the computed energy values...')
            bins = np.linspace(2, 15, 500)
            pyplot.hist(known, bins, alpha=0.5, label='known')
            pyplot.hist(unk, bins, alpha=0.5, label='unk')
            pyplot.legend(loc='upper right')
            pyplot.savefig(os.path.join(self.cfg.OUTPUT_DIR, 'energy.png'))
