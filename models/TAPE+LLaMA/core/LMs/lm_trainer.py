import torch
import numpy as np

from transformers import AutoTokenizer, AutoModel, TrainingArguments, Trainer, IntervalStrategy
from core.LMs.model import BertClassifier, BertClaInfModel
from core.data_utils.dataset import Dataset
from core.data_utils.load import load_data
from core.utils import init_path, time_logger
from sklearn.metrics import accuracy_score, f1_score



# def compute_metrics(p):
#     from sklearn.metrics import accuracy_score
#     pred, labels = p
#     pred = np.argmax(pred, axis=1)
#     accuracy = accuracy_score(y_true=labels, y_pred=pred)
#     return {"accuracy": accuracy}
def compute_metrics(p):
    pred, labels = p
    pred = np.argmax(pred, axis=1)  # Get the predicted class by taking argmax on the logits

    accuracy = accuracy_score(y_true=labels, y_pred=pred)  # Accuracy score
    micro_f1 = f1_score(y_true=labels, y_pred=pred, average='micro')  # Micro F1 score
    macro_f1 = f1_score(y_true=labels, y_pred=pred, average='macro')  # Macro F1 score
    
    # Return all metrics
    return {
        "accuracy": accuracy,
        "micro_f1": micro_f1,
        "macro_f1": macro_f1  # Add Macro F1 to the returned dictionary
    }

class LMTrainer():
    def __init__(self, cfg):
        self.dataset_name = cfg.dataset
        self.seed = cfg.seed

        self.model_name = cfg.lm.model.name
        self.feat_shrink = cfg.lm.model.feat_shrink

        self.weight_decay = cfg.lm.train.weight_decay
        self.dropout = cfg.lm.train.dropout
        self.att_dropout = cfg.lm.train.att_dropout
        self.cla_dropout = cfg.lm.train.cla_dropout
        self.batch_size = cfg.lm.train.batch_size
        self.epochs = cfg.lm.train.epochs
        self.warmup_epochs = cfg.lm.train.warmup_epochs
        self.eval_patience = cfg.lm.train.eval_patience
        self.grad_acc_steps = cfg.lm.train.grad_acc_steps
        self.lr = cfg.lm.train.lr

        self.use_gpt_str = "2" if cfg.lm.train.use_gpt else ""
        self.output_dir = f'output/{self.dataset_name}{self.use_gpt_str}/{self.model_name}-seed{self.seed}'
        self.ckpt_dir = f'prt_lm/{self.dataset_name}{self.use_gpt_str}/{self.model_name}-seed{self.seed}'

        # Preprocess data数据加载和预处理
        data, num_classes, text = load_data(
            dataset=self.dataset_name, use_text=True, use_gpt=cfg.lm.train.use_gpt, seed=self.seed)
        self.data = data
        self.num_nodes = data.y.size(0)
        self.n_labels = num_classes

        # if len(self.data.train_mask)== 10:
        #     self.data.train_mask = self.data.train_mask[0]
        #     self.data.val_mask = self.data.val_mask[0]
        #     self.data.test_mask = self.data.test_mask[0]
        # 加载分词器，对文本数据标记化，处理的是摘要数据
        # tokenizer = AutoTokenizer.from_pretrained(self.model_name)
        # tokenizer = AutoTokenizer.from_pretrained("sentence-transformers/multi-qa-distilbert-cos-v1")************更改
        # 将文本转换为bert可以理解的序列
        tokenizer = AutoTokenizer.from_pretrained("/mnt/data/zch/glbench/models/enhancer/TAPE/multi-qa-distilbert-cos-v1")
        # tokenizer = AutoTokenizer.from_pretrained("/mnt/data/zch/projects/models/enhancer/TAPE/multi-qa-distilbert-cos-v1", local_files_only=True)

        
        if type(text)!=list:
            text = text.tolist()
        X = tokenizer(text, padding=True, truncation=True, max_length=512)
        # Dataset 类将文本数据和标签封装为一个数据集对象
        dataset = Dataset(X, data.y.tolist())
        self.inf_dataset = dataset
        # 对训练数据的划分
        self.train_dataset = torch.utils.data.Subset(
            dataset, self.data.train_mask.nonzero().squeeze().tolist())
        self.val_dataset = torch.utils.data.Subset(
            dataset, self.data.val_mask.nonzero().squeeze().tolist())
        self.test_dataset = torch.utils.data.Subset(
            dataset, self.data.test_mask.nonzero().squeeze().tolist())

        # Define pretrained tokenizer and model
        # 模型初始化，预训练bert模型
        # bert_model = AutoModel.from_pretrained(self.model_name)
        # bert_model = AutoModel.from_pretrained("sentence-transformers/multi-qa-distilbert-cos-v1")************更改
        bert_model = AutoModel.from_pretrained("/mnt/data/zch/glbench/models/enhancer/TAPE/multi-qa-distilbert-cos-v1")
        # 自定义分类器，封装了bert_model,用于将bert的输出映射到分类标签空间
        self.model = BertClassifier(bert_model,
                                    n_labels=self.n_labels,
                                    feat_shrink=self.feat_shrink)# 是否降维

        # prev_ckpt = f'prt_lm/{self.dataset_name}/{self.model_name}.ckpt'
        # if self.use_gpt_str and os.path.exists(prev_ckpt):
        #     print("Initialize using previous ckpt...")
        #     self.model.load_state_dict(torch.load(prev_ckpt))

        self.model.config.dropout = self.dropout
        self.model.config.attention_dropout = self.att_dropout

        trainable_params = sum(p.numel()
                               for p in self.model.parameters() if p.requires_grad)
        print(f"\nNumber of parameters: {trainable_params}")

    @time_logger
    def train(self):
        # Define training parameters
        eq_batch_size = self.batch_size * 4
        train_steps = self.num_nodes // eq_batch_size + 1
        eval_steps = self.eval_patience // eq_batch_size
        warmup_steps = int(self.warmup_epochs * train_steps)

        # Define Trainer
        args = TrainingArguments(
            output_dir=self.output_dir,
            do_train=True,
            do_eval=True,
            eval_steps=eval_steps,
            evaluation_strategy=IntervalStrategy.STEPS,
            save_steps=eval_steps,
            learning_rate=self.lr,
            weight_decay=self.weight_decay,
            save_total_limit=1,
            load_best_model_at_end=True,
            gradient_accumulation_steps=self.grad_acc_steps,
            per_device_train_batch_size=self.batch_size,
            per_device_eval_batch_size=self.batch_size*8,
            warmup_steps=warmup_steps,
            num_train_epochs=self.epochs,
            dataloader_num_workers=1,
            fp16=True,
            dataloader_drop_last=False,
        )
        #训练过程
        self.trainer = Trainer(
            model=self.model,
            args=args,
            train_dataset=self.train_dataset,
            eval_dataset=self.val_dataset,
            compute_metrics=compute_metrics,
            # callbacks=[EarlyStoppingCallback(early_stopping_patience=3)],
        )

        # Train pre-trained model
        self.trainer.train()
        torch.save(self.model.state_dict(), init_path(f"{self.ckpt_dir}.ckpt"))
        print(f'LM saved to {self.ckpt_dir}.ckpt')

    #评估和保存
    @time_logger
    @torch.no_grad()
    def eval_and_save(self):
        emb = np.memmap(init_path(f"{self.ckpt_dir}.emb"),
                        dtype=np.float16,
                        mode='w+',
                        shape=(self.num_nodes, self.feat_shrink if self.feat_shrink else 768))
        pred = np.memmap(init_path(f"{self.ckpt_dir}.pred"),
                        dtype=np.float16,
                        mode='w+',
                        shape=(self.num_nodes, self.n_labels))

        inf_model = BertClaInfModel(
            self.model, emb, pred, feat_shrink=self.feat_shrink)
        inf_model.eval()
        
        inference_args = TrainingArguments(
            output_dir=self.output_dir,
            do_train=False,
            do_predict=True,
            per_device_eval_batch_size=self.batch_size*8,
            dataloader_drop_last=False,
            dataloader_num_workers=1,
            fp16_full_eval=True,
        )

        trainer = Trainer(model=inf_model, args=inference_args)
        trainer.predict(self.inf_dataset)
        
        if "ogbn" in self.dataset_name:
            from ogb.nodeproppred import Evaluator
            _evaluator = Evaluator(name=self.dataset_name)
        else:
            from core.GNNs.gnn_utils import Evaluator
            _evaluator = Evaluator(name=self.dataset_name)

        def evaluator(preds, labels):
            return _evaluator.eval({
                "y_true": torch.tensor(labels).view(-1, 1),
                "y_pred": torch.tensor(preds).view(-1, 1),
            })["acc"]

        def eval(x):
            return evaluator(np.argmax(pred[x], -1), self.data.y[x])

        # 计算训练集、验证集和测试集的准确率
        train_acc = eval(self.data.train_mask)
        val_acc = eval(self.data.val_mask)
        test_acc = eval(self.data.test_mask)

        # 计算宏观 F1 和微观 F1
        train_macro_f1 = f1_score(self.data.y[self.data.train_mask], np.argmax(pred[self.data.train_mask], -1), average='macro')
        val_macro_f1 = f1_score(self.data.y[self.data.val_mask], np.argmax(pred[self.data.val_mask], -1), average='macro')
        test_macro_f1 = f1_score(self.data.y[self.data.test_mask], np.argmax(pred[self.data.test_mask], -1), average='macro')

        train_micro_f1 = f1_score(self.data.y[self.data.train_mask], np.argmax(pred[self.data.train_mask], -1), average='micro')
        val_micro_f1 = f1_score(self.data.y[self.data.val_mask], np.argmax(pred[self.data.val_mask], -1), average='micro')
        test_micro_f1 = f1_score(self.data.y[self.data.test_mask], np.argmax(pred[self.data.test_mask], -1), average='micro')

        # 打印结果，包括 F1 分数
        print(
            f'[LM] TrainAcc: {train_acc:.4f}, ValAcc: {val_acc:.4f}, TestAcc: {test_acc:.4f}, '
            f'TrainMacroF1: {train_macro_f1:.4f}, ValMacroF1: {val_macro_f1:.4f}, TestMacroF1: {test_macro_f1:.4f}, '
            f'TrainMicroF1: {train_micro_f1:.4f}, ValMicroF1: {val_micro_f1:.4f}, TestMicroF1: {test_micro_f1:.4f}\n'
        )
        
        return {
            'TrainAcc': train_acc,'ValAcc': val_acc,'TestAcc': test_acc,
            'TrainMacroF1': train_macro_f1,'ValMacroF1': val_macro_f1,'TestMacroF1': test_macro_f1,
            'TrainMicroF1': train_micro_f1,'ValMicroF1': val_micro_f1,'TestMicroF1': test_micro_f1
        }
