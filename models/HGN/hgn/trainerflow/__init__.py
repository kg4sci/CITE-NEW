import importlib
from .base_flow import BaseFlow
from abc import ABC     

FLOW_REGISTRY = {}


def register_flow(name):
    
    """
    New flow can be added to openhgnn with the :func:`register_flow`
    function decorator.

    For example::

        @register_task('node_classification')
        class NodeClassification(BaseFlow):
            (...)

    Args:
        name (str): the name of the flows
    """

    def register_flow_cls(cls):
        if name in FLOW_REGISTRY:
            raise ValueError("Cannot register duplicate flow ({})".format(name))
        if not issubclass(cls, (BaseFlow,ABC)):
            raise ValueError("Flow ({}: {}) must extend BaseFlow or ABC".format(name, cls.__name__))
        FLOW_REGISTRY[name] = cls
        return cls

    return register_flow_cls


def try_import_flow(flow):
    if flow not in FLOW_REGISTRY:
        if flow in SUPPORTED_FLOWS:
            importlib.import_module(SUPPORTED_FLOWS[flow])
        else:
            print(f"Failed to import {flow} flows.")
            return False
    return True


def build_flow(args, flow_name):
    
    if not try_import_flow(flow_name):
        exit(1)
    return FLOW_REGISTRY[flow_name](args)


SUPPORTED_FLOWS = {
###########     add trainer_flow here. 【register name】 ： 【class name】

    
##########
    
    'node_classification': 'openhgnn.trainerflow.node_classification',
    'node_classification_ac': 'openhgnn.trainerflow.node_classfication_ac',
    
    'link_prediction': 'openhgnn.trainerflow.link_prediction',
    
    'hgttrainer': 'openhgnn.trainerflow.hgt_trainer',
    'nshetrainer': 'openhgnn.trainerflow.nshe_trainer',
    
    'han_nc_trainer': 'openhgnn.trainerflow.HANNodeClassification',
    'han_lp_trainer': 'openhgnn.trainerflow.HANLinkPrediction',
    
}

######      add trainer_flow here

##########

from .node_classification import NodeClassification
from .link_prediction import LinkPrediction

from .hgt_trainer import HGTTrainer

from .han_trainer import HANNodeClassification
from .han_trainer import HANLinkPrediction

from .node_classification_ac import NodeClassificationAC

#   don't add here
__all__ = [
    'BaseFlow',
    'NodeClassification',
    'LinkPrediction',
    
    'HANNodeClassification',
    'HANLinkPrediction',
    
]
classes = __all__
