import learn2learn.data as l2ldata
from wilds.datasets.wilds_dataset import WILDSSubset
import numpy as np
import torch




# """
# WildsMetaDataset
# modified MetaDataset that takes in a WILDSSubset
# """
# class WildsMetaDataset(l2ldata.MetaDataset):
#     def __init__(self, dataset: WILDSSubset, grouper):
#         assert isinstance(dataset, WILDSSubset)
#         self.dataset = dataset
#         self.groups, self.group_counts = grouper.metadata_to_group(
#             dataset.metadata_array,
#             return_counts=True
#         )
#         self.n_groups = len(self.group_counts)
#         self.create_bookkeeping(
#             labels_to_indices=None,
#             indices_to_labels=dict(zip(iter(dataset.y_array), iter(dataset.y_array)))
#         )


# """
# GroupTaskTransform 
# Task Transform for l2l MetaDataset that divides up tasks by groups
# """
# class GroupTaskTransform(l2ldata.transforms.TaskTransform):
#     def __init__(self, metadataset: WildsMetaDataset, n=2):
#         self.metadataset = metadataset
#         assert n <= metadataset.n_groups
#         self.n = n

#     def new_task(self):
#         task_description = []
#         groups_in_task = np.random.choice(
#             np.arange(self.metadataset.n_groups),
#             size=self.n,
#             replace=False
#         )
#         for g in groups_in_task:
#             idx = torch.nonzero(self.metadataset.groups == g)
#             task_description = task_description + [l2ldata.task_dataset.DataDescription(i) for i in idx]
#         return task_description
    
#     def __call__(self, task_description: []):
#         if task_description is None:
#             return self.new_task()
#         result = []
#         set_groups = set()
#         for dd in task_description:
#             set_groups.add(self.groups[dd.index])
#         task_groups = list(set_groups)
#         task_groups = np.random.choice(
#             task_groups,
#             size=self.n,
#             replace=False
#         )
#         for dd in task_description:
#             if self.groups[dd.index] in task_groups:
#                 result.append(dd)
#         return result




