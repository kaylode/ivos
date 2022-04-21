from theseus.semantic.augmentations.resampler import ScipyResample, ItkResample



zoom_image, zoom_factor = ScipyResample.resample_to_size(raw_image, self.cfg.COARSE_MODEL.INPUT_SIZE)
npy_image, _ = ScipyResample.resample_to_size(npy_image, out_size)
npy_mask, _ = ScipyResample.resample_mask_to_size(npy_mask, out_size,
                                                    num_label=np.max(np.array(self.label)))