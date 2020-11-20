# coding: utf-8

import tensorflow as tf


class TCNNConfig(object):
    """CNN配置参数"""
    embedding_dim = 256  # 词向量维度
    seq_length = 50  # 序列长度
    num_classes = 10  # 类别数
    num_filters = 256  # 卷积核数目
    kernel_size = 5  # 卷积核尺寸
    vocab_size = 5000  # 词汇表达小

    hidden_dim = 128  # 全连接层神经元

    dropout_keep_prob = 0.5  # dropout保留比例
    learning_rate = 1e-3  # 学习率

    batch_size = 64  # 每批训练大小
    num_epochs = 200  # 总迭代轮次

    print_per_batch = 10  # 每多少轮输出一次结果
    save_per_batch = 10  # 每多少轮存入tensorboard
   
        


class TextCNN(object):
    """文本分类，CNN模型"""

    def __init__(self, config):
        self.config = config

        # 三个待输入的数据
        self.input_x = tf.placeholder(tf.int32, [None, self.config.seq_length], name='input_x')
        self.input_y = tf.placeholder(tf.float32, [None, self.config.num_classes], name='input_y')
        self.keep_prob = tf.placeholder(tf.float32, name='keep_prob')
        self.regularizer = tf.contrib.layers.l2_regularizer(scale=0.0001)
        self.cnn()

    def cnn(self):
        """CNN模型"""
        # 词向量映射
        with tf.device('/cpu:0'):
            embedding = tf.get_variable('embedding', [self.config.vocab_size, self.config.embedding_dim])
            embedding_inputs = tf.nn.embedding_lookup(embedding, self.input_x)

        with tf.name_scope("cnn"):
            # CNN layer
            conv = tf.layers.conv1d(embedding_inputs, self.config.num_filters, self.config.kernel_size, name='conv', kernel_regularizer=self.regularizer)
            # global max pooling layer
            gmp = tf.reduce_max(conv, reduction_indices=[1], name='gmp')

        with tf.name_scope("score"):
            # 全连接层，后面接dropout以及relu激活
            fc = tf.layers.dense(gmp, self.config.hidden_dim, name='fc1',kernel_regularizer=self.regularizer)
            fc = tf.contrib.layers.dropout(fc, self.keep_prob)
            fc = tf.nn.relu(fc)

            # 分类器
            self.logits = tf.layers.dense(fc, self.config.num_classes, name='fc2',kernel_regularizer=self.regularizer)
            # self.logits = tf.layers.batch_normalization(self.logits, training=False)
            self.y_pred_cls = tf.argmax(tf.nn.softmax(self.logits), 1)# 预测类别
            self.score = tf.nn.softmax(self.logits) # 直接softmax
            max_num = tf.reduce_max(self.logits, axis=1)
            min_num = tf.reduce_min(self.logits)
            # 把logits全部变成正值，再除以他们的和，在0-1之间
            self.logits1 = (self.logits - min_num)/tf.reduce_sum(self.logits - min_num)
            # 对该比率softmax
            # self.score1 = tf.nn.softmax(logits1)
            # # 对logits1扩大， 放缩到 -5 到 5  
            # logits2 = logits1 * 10.0 - 5.0
            # # 对放大后的结果softmax
            # self.score2 = tf.nn.softmax(logits2)
            self.score3 = tf.sigmoid(self.logits1 * 5.0)
            

        with tf.name_scope("optimize"):
            # 损失函数，交叉熵
            cross_entropy = tf.nn.softmax_cross_entropy_with_logits(logits=self.logits, labels=self.input_y)
            self.loss = tf.reduce_mean(cross_entropy)
            #tf.add_to_collection('losses', self.loss) 
            #tf.add_to_collection('losses',tf.losses.get_regularization_losses())
            reg_losses = tf.get_collection(tf.GraphKeys.REGULARIZATION_LOSSES)
            #self.loss += tf.add_n(reg_losses)
            # 优化器
            self.optim = tf.train.AdamOptimizer(learning_rate=self.config.learning_rate).minimize(self.loss)
      
        with tf.name_scope("accuracy"):
            # 准确率
            correct_pred = tf.equal(tf.argmax(self.input_y, 1), self.y_pred_cls)
            self.acc = tf.reduce_mean(tf.cast(correct_pred, tf.float32))