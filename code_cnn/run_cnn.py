# 寻找最好的数据集
# 取正常文本数量为800
# 多次随机取800正常文本，寻找准确度最大的训练集

# 该实验对于训练集进行采样，正常类别个数在300，500,800,1000,2000,...20000
# 选取最佳的正常类别个数

#!/usr/bin/python
# -*- coding: utf-8 -*-

from __future__ import print_function

import os
import sys
import time
from datetime import timedelta
import random
import numpy as np
import tensorflow as tf
from sklearn import metrics

from cnn_model import TCNNConfig, TextCNN
from data_loader import read_vocab, read_category, batch_iter, process_file, build_vocab

os.environ["CUDA_VISIBLE_DEVICES"] = "0,1,2,3"

def get_time_dif(start_time):
    """获取已使用时间"""
    end_time = time.time()
    time_dif = end_time - start_time
    return timedelta(seconds=int(round(time_dif)))


def feed_data(x_batch, y_batch, keep_prob):
    feed_dict = {
        model.input_x: x_batch,
        model.input_y: y_batch,
        model.keep_prob: keep_prob
    }
    return feed_dict


def evaluate(sess, x_, y_):
    """评估在某一数据上的准确率和损失"""
    data_len = len(x_)
    batch_eval = batch_iter(x_, y_, 128)
    total_loss = 0.0
    total_acc = 0.0
    for x_batch, y_batch in batch_eval:
        batch_len = len(x_batch)
        feed_dict = feed_data(x_batch, y_batch, 1)
        loss, acc = sess.run([model.loss, model.acc], feed_dict=feed_dict)
        total_loss += loss * batch_len
        total_acc += acc * batch_len

    return total_loss / data_len, total_acc / data_len


def train():
    print("Configuring TensorBoard and Saver...")
    # 配置 Tensorboard，重新训练时，请将tensorboard文件夹删除，不然图会覆盖
    tensorboard_dir = 'tensorboard/textcnn'
    if not os.path.exists(tensorboard_dir):
        os.makedirs(tensorboard_dir)

    tf.summary.scalar("loss", model.loss)
    tf.summary.scalar("accuracy", model.acc)
    merged_summary = tf.summary.merge_all()
    writer = tf.summary.FileWriter(tensorboard_dir)

    # 配置 Saver
    saver = tf.train.Saver()
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
    if not os.path.exists(save_dir_bak):
        os.makedirs(save_dir_bak)
    print("Loading training and validation data...")
    # 载入训练集与验证集
    start_time = time.time()
    x_train, y_train = process_file(train_dir, word_to_id, cat_to_id, config.seq_length)
    # 不需要验证集
    # x_val, y_val = process_file(val_dir, word_to_id, cat_to_id, config.seq_length)
    x_test, y_test = process_file(test_dir, word_to_id, cat_to_id, config.seq_length)
    time_dif = get_time_dif(start_time)
    print("Time usage:", time_dif)

    # 创建session
    gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=0.1) 
    session = tf.Session(config=tf.ConfigProto(gpu_options=gpu_options))
    session.run(tf.global_variables_initializer())
    writer.add_graph(session.graph)

    print('Training and evaluating...')
    start_time = time.time()
    total_batch = 0  # 总批次
    # best_acc_val = 0.0  # 最佳验证集准确率
    best_acc_test = 0.0
    last_improved = 0  # 记录上一次提升批次
    require_improvement = 1500  # 如果超过1000轮未提升，提前结束训练

    flag = False
    for epoch in range(config.num_epochs):
        print('Epoch:', epoch + 1)
        batch_train = batch_iter(x_train, y_train, config.batch_size)
        for x_batch, y_batch in batch_train:
            feed_dict = feed_data(x_batch, y_batch, config.dropout_keep_prob)

            if total_batch % config.save_per_batch == 0:
                # 每多少轮次将训练结果写入tensorboard scalar
                s = session.run(merged_summary, feed_dict=feed_dict)
                writer.add_summary(s, total_batch)

            if total_batch % config.print_per_batch == 0:
                # 每多少轮次输出在训练集和验证集上的性能
                feed_dict[model.keep_prob] = 1.0
                loss_train, acc_train = session.run([model.loss, model.acc], feed_dict=feed_dict)
                # loss_val, acc_val = evaluate(session, x_val, y_val)  # todo
                loss_test, acc_test = evaluate(session, x_test, y_test)
                if acc_test > best_acc_test:
                    # 保存最好结果
                    best_acc_test = acc_test
                    last_improved = total_batch
                    saver.save(sess=session, save_path=save_path_bak)
                    improved_str = '*'
                else:
                    improved_str = ''

                time_dif = get_time_dif(start_time)
                msg = 'Iter: {0:>6}, Train Loss: {1:>6.2}, Train Acc: {2:>7.2%},' \
                      + ' Test Loss: {3:>6.2}, Test Acc: {4:>7.2%}, Time: {5} {6}'
                print(msg.format(total_batch, loss_train, acc_train, loss_test, acc_test, time_dif, improved_str))

            feed_dict[model.keep_prob] = config.dropout_keep_prob
            session.run(model.optim, feed_dict=feed_dict)  # 运行优化
            total_batch += 1

            if total_batch - last_improved > require_improvement:
                # 验证集正确率长期不提升，提前结束训练
                print("No optimization for a long time, auto-stopping...")
                flag = True
                break  # 跳出循环
        if flag:  # 同上
            break


def test():
    print("Loading test data...")
    start_time = time.time()
    x_test, y_test = process_file(test_dir, word_to_id, cat_to_id, config.seq_length)

    gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=0.1) 
    session = tf.Session(config=tf.ConfigProto(gpu_options=gpu_options))
    
    session.run(tf.global_variables_initializer())
    saver = tf.train.Saver()
    saver.restore(sess=session, save_path=save_path)  # 读取保存的模型

    print('Testing...')
    loss_test, acc_test = evaluate(session, x_test, y_test)
    msg = 'Test Loss: {0:>6.2}, Test Acc: {1:>7.2%}'
    print(msg.format(loss_test, acc_test))

    batch_size = 128
    data_len = len(x_test)
    num_batch = int((data_len - 1) / batch_size) + 1

    y_test_cls = np.argmax(y_test, 1)
    y_pred_cls = np.zeros(shape=len(x_test), dtype=np.int32)  # 保存预测结果
    for i in range(num_batch):  # 逐批次处理
        start_id = i * batch_size
        end_id = min((i + 1) * batch_size, data_len)
        feed_dict = {
            model.input_x: x_test[start_id:end_id],
            model.keep_prob: 1.0
        }
        y_pred_cls[start_id:end_id] = session.run(model.y_pred_cls, feed_dict=feed_dict)

    # Error print
    for i in range(len(y_pred_cls)):
        if y_pred_cls[i] != y_test_cls[i]:
            print('Predict:',categories[y_pred_cls[i]],'True:',categories[y_test_cls[i]])
            print('Sentence:',''.join([words[id] for id in x_test[i] if id != 0]))
            
    # 评估
    print("Precision, Recall and F1-Score...")
    print(metrics.classification_report(y_test_cls, y_pred_cls, target_names=categories))

    # 混淆矩阵
    print("Confusion Matrix...")
    cm = metrics.confusion_matrix(y_test_cls, y_pred_cls)
    print(cm)

    time_dif = get_time_dif(start_time)
    print("Time usage:", time_dif)

def get_train_data(num):
    id2label = {'0': '业务技能规范-混淆我司为贷款公司', '1': '业务技能规范-逼迫其通过非法途径筹集资金', '2': '违规惩处-冒用银行名义开展催收', '3': '违规惩处-将借款人信息公布在公开信息平台', 
    '4': '违规惩处-私自承诺减免', '5': '业务技能规范-拖车', '6': '业务技能规范-承认我方为高利贷、黑社会等', '7': '违规惩处-冒用行政、司法机关名义开展催收', '8': '违规惩处-不当言辞', 
    '9': '违规惩处-爆通讯录', '10': '客户未知意图', '11': '用语恰当', '12': '服务禁语', '13': '打断客户', '14': '安抚致歉', '15': '耐心倾听', '16': '正常文本'}
    id2text = {}
    with open(os.path.join(base_dir,'normal_file.txt')) as f:
        for line in f:
            line = line.strip().split('|,|')
            if line[0] not in id2text:
                id2text[line[0]] = [line[2]]
            else:
                if line[2] not in id2text[line[0]]:
                    id2text[line[0]].append(line[2])
    with open(os.path.join(base_dir,'bad_case_file.txt')) as f:
        for line in f:
            line = line.strip().split('|,|')
            if line[0] not in id2text:
                id2text[line[0]] = [line[1]]
            else:
                id2text[line[0]].append(line[1])
    with open(os.path.join(base_dir,'online_corpus.txt')) as f:
        for line in f:
            line = line.strip().split('|,|')
            if line[0] not in id2text:
                id2text[line[0]] = [line[2]]
            else:
                if line[2] not in id2text[line[0]]:
                    id2text[line[0]].append(line[2])
    fw = open(os.path.join(base_dir,'final_train.txt'),'w')
    lines = []
    for id,texts in id2text.items():
        if id!='16':
            t = random.choices(texts,k=num)
            for text in t:
                # lines.append(text+'\t'+id+'\t'+'意图'+id)
                lines.append(id+'|,|'+id2label[id]+'|,|'+text)
        else:
            t = texts
            for text in t:
                # lines.append(text+'\t'+id+'\t'+'正常文本')
                lines.append(id+'|,|'+id2label[id]+'|,|'+text)
    for line in lines:
        fw.write(line+'\n')
    fw.close()

if __name__ == '__main__':
        
    base_dir = '../data_qc_oversampling'
    # data_dir = os.path.join(base_dir, 'train_new.txt')
    train_dir = os.path.join(base_dir, 'final_train.txt')
    test_dir = os.path.join(base_dir, 'test.txt')
    # val_dir = os.path.join(base_dir, 'data.val.txt')
    vocab_dir = os.path.join(base_dir, 'vocab.txt')

    save_dir_bak = 'checkpoints/textcnn_bak'
    save_path_bak = os.path.join(save_dir_bak, 'best_validation')  # 最佳验证结果保存路径

    save_dir = 'checkpoints/textcnn'
    save_path = os.path.join(save_dir, 'best_validation')

    print('Configuring CNN model...')
   
    normal_num = [690] * 3
    max_acc = 0
    greatest_normal_num = 0
    for i in normal_num:
        get_train_data(i)
        config = TCNNConfig()
        if not os.path.exists(vocab_dir):  # 如果不存在词汇表，重建
            build_vocab(train_dir, vocab_dir, config.vocab_size)
        categories, cat_to_id = read_category(train_dir)
        words, word_to_id = read_vocab(vocab_dir)
        config.vocab_size = len(words)
        config.num_classes = len(categories)
        model = TextCNN(config)
        # 训练模型并保存到bak
        train()

        print("Loading test data...")
        start_time = time.time()
        x_test, y_test = process_file(test_dir, word_to_id, cat_to_id, config.seq_length)

        gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=0.1) 
        session = tf.Session(config=tf.ConfigProto(gpu_options=gpu_options))
        
        session.run(tf.global_variables_initializer())
        saver = tf.train.Saver()
        saver.restore(sess=session, save_path=save_path_bak)  # 读取保存的模型

        print('Testing...')
        loss_test, acc_test = evaluate(session, x_test, y_test)
        if acc_test > max_acc and loss_test < 5:
            print('GET GREATEST MODEL!')
            max_acc = acc_test
            greatest_normal_num = i
            saver.save(sess=session, save_path=save_path)

        tf.reset_default_graph()
    print('max_acc:',max_acc, 'class16:',greatest_normal_num)