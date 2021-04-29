from .cls_head import ClsHead
from .linear_head import LinearClsHead
from .supcon_head import SupConClsHead, DASupConClsHead
from .cluster_head import DASupClusterHead

__all__ = ['ClsHead', 'LinearClsHead', 'SupConClsHead', 'DASupConClsHead', 
           'DASupClusterHead']
