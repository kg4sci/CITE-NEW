# from openhgnn.dataset import build_dataset
from dataset import build_dataset
import logging

def test_my_dataset():
    # 构建你的数据集
    logger = logging.getLogger("my_logger")
    logger.setLevel(logging.INFO)
    # self.logger = args.logger
    dataset = build_dataset('my_custom_node_classification', task='node_classification',logger=logger)

    # 打印图的基本信息
    print("Graph:", dataset.g)
    print("目标节点类型:", dataset.category)
    print("类别数量:", dataset.num_classes)
    print("是否有特征:", dataset.has_feature)
    print("特征维度:", dataset.g.nodes[dataset.category].data['feature'].shape)
    print("训练集、验证集、测试集:")
    print(dataset.graph)
    # print("Train:", len(dataset.get_split.train_idx))
    # print("Valid:", len(dataset.get_split.valid_idx))
    # print("Test:", len(dataset.get_split.test_idx))

if __name__ == "__main__":
    test_my_dataset()
