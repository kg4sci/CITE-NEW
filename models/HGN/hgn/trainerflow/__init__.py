import importlib
from .base_flow import BaseFlow
from abc import ABC     

FLOW_REGISTRY = {}# 运行时保存已经注册或已经 import 并成功放入的 trainer flow


def register_flow(name):# 装饰器
    """
    当我们创建一个新的流程类，比如 class MyNewFlow(BaseFlow): ...，可以在它上面写
    @register_flow('my_new_flow')
    class MyNewFlow(BaseFlow):
        ...
    这样就会自动执行 register_flow_cls，把 'my_new_flow' -> MyNewFlow 存入 FLOW_REGISTRY
    
    """
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
    if flow not in FLOW_REGISTRY:#检查是否在FLOW_REGISTRY中
        if flow in SUPPORTED_FLOWS:
            importlib.import_module(SUPPORTED_FLOWS[flow])
        else:
            print(f"Failed to import {flow} flows.")
            return False
    return True


def build_flow(args, flow_name):
    """
    args: 是 Experiment 类中的 config 配置对象，包含了所有的配置信息，包括模型、任务类型、数据集等
    flow_name: 是指定的训练流程名称，例如 'MHGCN_trainer'、'BPHGNN_trainer' 等
    整个框架里最常见的入口就是 build_flow(self.config, trainerflow)。
    它先 try_import_flow(flow_name)；如果失败就 exit(1) 终止程序；
    成功后则在 FLOW_REGISTRY 里直接拿到对应的类，调用它的构造方法 (args) 来生成实例。
    最后返回这个实例给上层（例如 Experiment.run()），就可以调用它的 .train() 等方法
    """
    if not try_import_flow(flow_name):
        exit(1)
    return FLOW_REGISTRY[flow_name](args)


SUPPORTED_FLOWS = {
###########     add trainer_flow here. 【register name】 ： 【class name】

    'MHGCN_trainer':'openhgnn.trainerflow.MHGCN_trainer',
    'BPHGNN_trainer':'openhgnn.trainerflow.BPHGNN_trainer',    
    'HGMAE':'openhgnn.trainerflow.HGMAE_trainer',
    'hga_trainer':'openhgnn.trainerflow.HGATrainer',
    'rhine_trainer':'openhgnn.trainerflow.RHINETrainer',
    'FED_REC_trainer':'openhgnn.trainerflow.FED_Recommendation',
##########
    "coldstart_recommmendation": "openhgnn.trainerflow.coldstart_recommendation",
    'SIAN_trainer': 'openhgnn.trainerflow.SIAN_trainer',
    'entity_classification': 'openhgnn.trainerflow.entity_classification',
    'node_classification': 'openhgnn.trainerflow.node_classification',
    'node_classification_ac': 'openhgnn.trainerflow.node_classfication_ac',
    'distmult': 'openhgnn.trainerflow.dist_mult',
    'link_prediction': 'openhgnn.trainerflow.link_prediction',
    'recommendation': 'openhgnn.trainerflow.recommendation',
    'hetgnntrainer': 'openhgnn.trainerflow.hetgnn_trainer',
    'hgttrainer': 'openhgnn.trainerflow.hgt_trainer',
    'nshetrainer': 'openhgnn.trainerflow.nshe_trainer',
    'demo': 'openhgnn.trainerflow.demo',
    'kgcntrainer': 'openhgnn.trainerflow.kgcn_trainer',
    'HeGAN_trainer': 'openhgnn.trainerflow.HeGAN_trainer',
    'mp2vec_trainer': 'openhgnn.trainerflow.mp2vec_trainer',
    'herec_trainer': 'openhgnn.trainerflow.herec_trainer',
    'HeCo_trainer': 'openhgnn.trainerflow.HeCo_trainer',
    'DMGI_trainer': 'openhgnn.trainerflow.DMGI_trainer',
    'slicetrainer': 'openhgnn.trainerflow.slice_trainer',
    'hde_trainer': 'openhgnn.trainerflow.hde_trainer',
    'GATNE_trainer': 'openhgnn.trainerflow.GATNE_trainer',
    'TransX_trainer': 'openhgnn.trainerflow.TransX_trainer',
    'han_nc_trainer': 'openhgnn.trainerflow.HANNodeClassification',
    'han_lp_trainer': 'openhgnn.trainerflow.HANLinkPrediction',
    'RoHe_trainer': 'openhgnn.trainerflow.RoHe_trainer',
    'mg2vec_trainer': 'openhgnn.trainerflow.mg2vec_trainer',
    'DHNE_trainer': 'openhgnn.trainerflow.DHNE_trainer',
    'DiffMG_trainer': 'openhgnn.trainerflow.DiffMG_trainer',
    'MeiREC_trainer': 'openhgnn.trainerflow.MeiRec_trainer',
    'abnorm_event_detection': 'openhgnn.trainerflow.AbnormEventDetection',
    'SHGP_trainer': 'openhgnn.trainerflow.SHGP_trainer',
    'KGAT_trainer': 'openhgnn.trainerflow.KGAT_trainer',
    'DSSL_trainer': 'openhgnn.trainerflow.DSSL_trainer',
    'hgcltrainer': 'openhgnn.trainerflow.hgcl_trainer',
    'lightGCN_trainer': 'openhgnn.trainerflow.lightGCN_trainer',
    'KTN_trainer':'openhgnn.trainerflow.KTN_trainer',
    'SeHGNN_trainer': 'openhgnn.trainerflow.SeHGNN_trainer',
    'Grail_trainer': 'openhgnn.trainerflow.Grail_trainer',
    'ComPILE_trainer': 'openhgnn.trainerflow.ComPILE_trainer',
    'AdapropT_trainer': 'openhgnn.trainerflow.AdapropT_trainer',
    'AdapropI_trainer':'openhgnn.trainerflow.AdapropI_trainer',
    'LTE_trainer': 'openhgnn.trainerflow.LTE_trainer',
    'SACN_trainer': 'openhgnn.trainerflow.SACN_trainer',
    'ExpressGNN_trainer': 'openhgnn.trainerflow.ExpressGNN_trainer',
    'NBF_trainer':'openhgnn.trainerflow.NBF_trainer',
    'Ingram_Trainer' : 'openhgnn.trainerflow.Ingram_trainer',
    'DisenKGAT_trainer':'openhgnn.trainerflow.DisenKGAT_trainer',
    'RedGNN_trainer': 'openhgnn.trainerflow.RedGNN_trainer',
    'RedGNNT_trainer': 'openhgnn.trainerflow.RedGNNT_trainer',
    'HGPrompt':'openhgnn.trainerflow.HGPrompt_trainer',



}

######      add trainer_flow here
from .BPHGNN_trainer import BPHGNN_trainer
from .HGMAE_trainer import HGMAE_trainer
from .hga_trainer import HGATrainer
from .RHINE_trainer import RHINETrainer
from .FED_REC_trainer import *
##########
from .sian_trainer import SIAN_Trainer
from .hgcl_trainer import HGCLtrainer
from .node_classification import NodeClassification
from .link_prediction import LinkPrediction
from .recommendation import Recommendation
from .hetgnn_trainer import HetGNNTrainer
from .hgt_trainer import HGTTrainer
from .kgcn_trainer import KGCNTrainer
from .HeGAN_trainer import HeGANTrainer
from .mp2vec_trainer import Metapath2VecTrainer
from .herec_trainer import HERecTrainer
from .HeCo_trainer import HeCoTrainer
from .DMGI_trainer import DMGI_trainer
from .slice_trainer import SLiCETrainer
from .hde_trainer import hde_trainer
from .GATNE_trainer import GATNE
from .han_trainer import HANNodeClassification
from .han_trainer import HANLinkPrediction
from .RoHe_trainer import RoHeTrainer
from .mg2vec_trainer import Mg2vecTrainer
from .DHNE_trainer import DHNE_trainer
from .DiffMG_trainer import DiffMG_trainer
from .MeiRec_trainer import MeiRECTrainer
from .kgat_trainer import KGAT_Trainer
from .node_classification_ac import NodeClassificationAC
from .DSSL_trainer import DSSL_trainer
from .lightGCN_trainer import lightGCNTrainer
from .KTN_trainer import KTN_NodeClassification
from .SeHGNN_trainer import SeHGNNtrainer
from .Grail_trainer import GrailTrainer
from .ComPILE_trainer import ComPILETrainer
from .AdapropT_trainer import AdapropTTrainer
from .AdapropI_trainer import AdapropITrainer
from .LTE_trainer import LTETrainer
from .SACN_trainer import SACNTrainer
from .ExpressGNN_trainer import ExpressGNNTrainer
from .NBF_trainer import * 
from .Ingram_trainer import Ingram_Trainer
from .DisenKGAT_trainer import *
from .RedGNNT_trainer import RedGNNTTrainer
from .HGPrompt import HGPrompt_trainer



#   don't add here
__all__ = [
    'MHGCN_trainer',
    'BaseFlow',
    'NodeClassification',
    'LinkPrediction',
    'Recommendation',
    'HetGNNTrainer',
    'HGTTrainer',
    'KGCNTrainer',
    'HeGANTrainer',
    'Metapath2VecTrainer',
    'HERecTrainer',
    'HeCoTrainer',
    'DMGI_trainer',
    'SLiCETrainer',
    'hde_trainer',
    'SIAN_Trainer',
    'GATNE',
    'HANNodeClassification',
    'HANLinkPrediction',
    'Mg2vecTrainer',
    'DHNE_trainer',
    'DiffMG_trainer',
    'MeiRECTrainer',
    'KGAT_Trainer',
    'DSSL_trainer',
    'HGCLtrainer',
    'lightGCNTrainer',
    'KTN_NodeClassification',
    'SeHGNNtrainer',
    'GrailTrainer',
    'ComPILETrainer',
    'AdapropTTrainer',
    'AdapropITrainer',
    'LTETrainer',
    'SACNTrainer',
    'ExpressGNNTrainer',
    'Ingram_trainer',
    'RHINETrainer',

]
classes = __all__
