from typing import List
import torch
import numpy as np

class Referencer:
    """
    Masks should be in (T, H, W) shape
    """
    def __init__(self) -> None:
        pass

    def search_reference(self, mask: np.ndarray, strategy: str ="non-empty"):
        """
        vol_mask: argmaxed segmentation mask. (T, H, W)
        """
        assert strategy in ['non-empty', 'most-classes', 'random', 'least-uncertainty'], "Wrong strategy chosen"

        if strategy == 'non-empty':
            return self._get_non_empty(mask)
        if strategy == 'most-classes':
            return self._get_most_classes(mask)
        if strategy == 'random':
            return self._get_random_slices(mask)
        if strategy == 'least-uncertainty':
            return self._get_least_uncertainty(mask)
        if strategy == 'largest-area':
            return self._get_all_classes(mask)


    def _get_most_classes(self, mask: np.ndarray):
        """
        Search for guide frames, in which most classes are presented
        """
        num_slices = mask.shape[0]
        candidate_indices = []
        max_possible_number_of_classes = 0
        for frame_idx in range(num_slices):
            num_classes = len(np.unique(mask[frame_idx, :, :]))
            if num_classes == max_possible_number_of_classes:
                candidate_indices.append(frame_idx)
            elif num_classes > max_possible_number_of_classes:
                max_possible_number_of_classes = num_classes
                candidate_indices = [frame_idx]

        return candidate_indices, max_possible_number_of_classes

    def _get_all_classes(self, mask: np.ndarray):
        """
        Search for guide frames, in which all classes are presented
        """

        num_slices = mask.shape[0]

        available_classes = np.unique(mask)
        class_dict = {
            int(k): [] for k in available_classes
        }
        for frame_idx in range(num_slices):
            current_classes = np.unique(mask[frame_idx, :, :])
            for cl in current_classes:
                class_dict[int(cl)].append({
                    'index': frame_idx,
                    'area': np.sum(mask[frame_idx, :, :] == cl)
                })

        candidate_indices = []
        for k in available_classes:
            sorted_class_dict_by_class_area = sorted(class_dict[k], key=lambda d: d['area'], reverse=True) 
            candidate_indices.append(sorted_class_dict_by_class_area[0])

        candidate_indices = list(set(candidate_indices)).sort()
        return candidate_indices

    def _get_non_empty(self, mask: np.ndarray):
        """
        Search for non-empty mask slices
        """
        num_slices = mask.shape[0]
        candidate_indices = []
        for frame_idx in range(num_slices):
            num_classes = len(np.unique(mask[frame_idx, :, :]))
            if num_classes > 0:
                candidate_indices.append(frame_idx)
        return candidate_indices
    
    def _get_random_slices(self, mask: np.ndarray, num_slices: int=1):
        """
        Search for non-empty mask slices
        """
        candidate_indices = np.random.choice(range(mask.shape[0]), size=num_slices, replace=False)

        if not isinstance(candidate_indices, list):
            candidate_indices = [candidate_indices]
        return candidate_indices

    def _get_least_uncertainty(self, mask):
        """
        Active learning
        """
        pass


    def find_propagation_range(self, indexes: List, length: int):
        """
        Produce propagation range for propagation module
        list of (reference id, propagation length)
        """

        if len(indexes) == 1:
            prop_range = \
                [(indexes[0], 0)] \
                + [(indexes[0], length)] 
        else:
            prop_range = [
                (indexes[i], indexes[i+1])
                for i in range(len(indexes)-1)
            ]

            prop_range = \
                [(min(indexes), 0)] \
                + prop_range \
                + [(max(indexes), length)] 

        return prop_range


    def search_reference_and_pack(self, vol_mask: np.ndarray, global_indices: List[int], pad_length: int, strategy: str="all"):
        """
        vol_mask: argmaxed segmentation mask. (T//sr, H, W)
        Return
            pad_length: int, return reference masks in targeted length with padding
            propagation range
        """
        
        # Generate reference frame, contains most suitable annotation masks 
        candidates_local_indices = self.search_reference(vol_mask, strategy=strategy)

        # Convert local to global indexes
        candidates_global_indices = [global_indices[i] for i in candidates_local_indices]
        global_to_local = {
          k:v for k, v in zip(candidates_global_indices, candidates_local_indices)
        }

        # Padding
        masks = []
        for local_idx, global_idx in enumerate(range(pad_length)):
            if global_idx in candidates_global_indices:
                masks.append(vol_mask[global_to_local[global_idx]])
            else:
                masks.append(np.zeros_like(vol_mask[0]))
        
        # To tensor
        masks = np.stack(masks, 0)
        masks = torch.from_numpy(masks)

        return masks, candidates_global_indices