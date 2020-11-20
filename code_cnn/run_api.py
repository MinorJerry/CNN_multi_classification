from __future__ import print_function

import os
import sys
import time
from datetime import timedelta
import re
import numpy as np
import tensorflow as tf
from sklearn import metrics

from cnn_model import TCNNConfig, TextCNN
from data_loader import read_vocab, read_category, batch_iter, process_file, build_vocab
import tensorflow.keras as kr
os.environ["CUDA_VISIBLE_DEVICES"] = "-1"

base_dir = '../data_qc_oversampling'
train_dir = os.path.join(base_dir, 'final_train.txt')
vocab_dir = os.path.join(base_dir, 'vocab.txt')

save_dir = 'checkpoints/textcnn'
save_path = os.path.join(save_dir, 'best_validation')  # 最佳验证结果保存路径

config = TCNNConfig()
"""
词典一旦变化会极大影响到效果，因此词典不可变
if not os.path.exists(vocab_dir):  # 如果不存在词汇表，重建
    build_vocab(train_dir, vocab_dir, config.vocab_size)
"""
categories, cat_to_id = read_category(train_dir)
words, word_to_id = read_vocab(vocab_dir)
config.vocab_size = len(words)
config.num_classes = len(categories)
model = TextCNN(config)

gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=0.1) 
session = tf.Session(config=tf.ConfigProto(gpu_options=gpu_options))

session.run(tf.global_variables_initializer())
saver = tf.train.Saver()
saver.restore(sess=session, save_path=save_path)  # 读取保存的模型

def test(sentence):
    data_id = []
    data_id.append([word_to_id[x] for x in sentence if x in word_to_id])
    x_test = kr.preprocessing.sequence.pad_sequences(data_id, config.seq_length)
    feed_dict = {
            model.input_x: x_test,
            model.keep_prob: 1.0
        }
    y_pred = session.run(model.y_pred_cls, feed_dict = feed_dict)
    logits, score, score3 = session.run([model.logits, model.score, model.score3] , feed_dict = feed_dict)
    print('logits:',logits)
    print('score:',score[0])
    # print('score1:',score1[0])
    # print('score2:',score2[0])
    print('score3:',score3[0])
    #score2 = session.run(model.score2, feed_dict = feed_dict)
    #print(score2[0][y_pred[0]])
    # print(categories[y_pred[0]])
    return categories[y_pred[0]], score[0][y_pred[0]]

def get_regular_result(pred_label, s):
    if pred_label == '16':
        for key_word in ['妈的','老赖','屁用','死猪不怕开水烫','垃圾','我操','做梦']:#some_key_words['8']:
            if key_word in s:
                pred_label = '8'
                break 
    if pred_label == '5':
        if re.search(r'(没有|没|不)[^，。]*[拖收拉拽].*车',s):
            pred_label = '16'
    elif (re.search(r'收[^，。]*车(?!(的话|吗))',s) or \
        re.search(r'车[^，。]*拖了(?!(的话|吗))',s) or \
        re.search(r'车[^，。他她]*收了(?!(的话|吗))',s) or \
        re.search(r'拖[^，。]?车了',s) or \
        re.search(r'车[^，。他她]*收走(?!(的话|吗))',s) or \
        re.search(r'车[^不]*会.*收走',s)) and \
        re.search(r'(没有|没|不是|不会)[^，。]*收.*车',s) == None:
        pred_label = '5'  

    if pred_label == '16' or pred_label == '4':
        if (re.search(r'^[^申请]*(减掉|降低|减少|免|少交|减免).*(违约金|本金|利息|钱|一些|一部分|一点)[^，]*[^吗]',s) or \
            re.search(r'^[^申请]*(违约金|本金|利息|钱).*(减掉|降低|减少|免|少交|减免)',s) or \
            re.search(r'^[^申请]*(减掉|降低|减少|免|少交|减免).*(违约金|本金|利息|钱)',s)) and \
            re.search(r'(怎么|如何|怎样|咋|没有办法|不|应该可以|试着)[^，。]*(减掉|降低|减少|免|少交|减免)',s) == None and \
            re.search(r'(减掉|降低|减少|免|少交|减免).*(违约金|本金|利息|钱)[^，。]*吗',s) == None and \
            re.search(r'(减掉|降低|减少|免|少交|减免)*(不了|不来|不掉)',s) == None and \
            '无法' not in s and '没办法' not in s and '没法' not in s and ('不能' not in s or '能不能' in s) and '申请' not in s \
            and '报备' not in s and '谈一下' not in s:
            pred_label = '4'
        else:
            pred_label = '16'
    if pred_label == '16':
        if (
            re.search(r'(去带|去贷|去借|看看|找找|想想).*(别的|其他|另外).*贷款',s) or \
            re.search(r'(去带|去贷|去借).*(信用卡|支付宝|借呗|花呗|京东|白条)',s) or \
            re.search(r'(信用卡|支付宝|借呗|花呗|京东|白条).*(找钱|来补|贷款|先还|借钱|取钱|周转|调整|转换|套现|套钱)',s) or \
            re.search(r'以贷养贷',s) or re.search(r'美团.*贷款',s) or \
            re.search(r'(信用卡|支付宝|借呗|花呗|京东|白条|银行卡).*(带钱|贷钱|去带|去贷|去借|借啊|借吗)',s)
            ) and re.search(r'(怎能|怎么能|不能|别去|不要|如果|禁止)',s) == None:
            pred_label = '1'

    if re.search(r'(爆|公布|发|挂|公开).*(照片|个人信息)',s) or \
        re.search(r'(照片|个人信息).*(爆|公布|发|挂|公开)',s):
        pred_label = '3'

    if re.search(r'(家里|家人|爸妈|亲戚|朋友|同事|儿子|女儿|儿女|老婆|什么人)[^，。]*(打一遍|会知道)',s):
        pred_label = '9'

    if pred_label in ['10', '12', '13', '14', '15']:
        if re.search(r'(对不起|歉意|抱歉|别担心|我们的问题|我们的原因)',s):
            pred_label = '14'
        elif re.search(r'(听我说|别讲话|别说)',s):
            pred_label = '12'
        elif re.search(r'(耐心听|我在听|慢慢说)',s):
            pred_label = '15'
        if re.search(r'(打断一下|插个嘴|先听我说)',s):
            pred_label = '13'

    return pred_label

if __name__ == "__main__":
    print('Testing...')
    
    s = '我要打爆你的电话'
    pred_label,score = test(s)
    # pred_label = get_regular_result(pred_label,s)
    print(pred_label,score)

    # 文件测试方法
    """
    sentences = []
    labels = []
    f = open('../data_qc_oversampling/test.txt')
    for line in f:
        line = line.strip().split('\t')
        sentences.append(line[0])
        labels.append(line[1])
    f.close()
    count = 0
    for i in range(len(sentences)):
        start = time.time()
        s = sentences[i]
        pred_label,score = test(s)
        
        true_label = labels[i]
        pred_label = get_regular_result(pred_label, s)
        if true_label == pred_label: 
            count += 1
        else: 
            print('Predict:',pred_label,'True',true_label)
            print('Sentence',sentences[i][-50:])
        end = time.time()
        # print(end-start)
    print(count/len(sentences))
    """
