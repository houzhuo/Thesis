#coding=utf-8

from numpy import *
import jieba
import re
import codecs
import time

__author__ = 'houzhuo.cs1994'
import sys
reload(sys)
sys.setdefaultencoding('utf-8')



# def ifutf8():
#     if isinstance(input, unicode):
#         temp2 = input.encode('utf8')
#     else:
#         temp2 = input


def filter_name_link(content):
    pattern = re.compile(r"(@.*?\s)|(http://.*?\s)")
    no_name_link = re.sub(pattern, "", content)
    return no_name_link


def filter_data():
    i = 0
    j = 0
    new_file = file('C:\Users\Zhuo\Documents\weibo_base_new.txt', 'w')
    for line in open('C:\Users\Zhuo\Documents\weibo_base.txt', 'r'):
        line = line.strip('\n').split('\001')
        if len(line) > 2:
            new_file.writelines('\001' + line[1] + '\001' + line[2] + '\n')
            i += 1
        else:
            j += 1
    new_file.close()


def load_data():
    new_file = file('C:\Users\Zhuo\Documents\weibo_base_filter.txt', 'w')
    for line in open('C:\Users\Zhuo\Documents\weibo_base_new.txt', 'r'):
        line = line.strip('\n').split('\001')
        no_name_link = filter_name_link('\001' + line[1] + '\001' + line[2])
        new_file.writelines(no_name_link + '\n')
    new_file.close()


def generate_neg_pos():
    n = 0
    p = 0
    j = 0
    new_file_pos = file('C:\Users\Zhuo\Documents\weibo_base_filter_pos.txt', 'w')
    new_file_neg = file('C:\Users\Zhuo\Documents\weibo_base_filter_neg.txt', 'w')
    for line in open('C:\Users\Zhuo\Documents\weibo_base_filter.txt', 'r'):
        line = line.strip('\n').split('\001')
        if len(line) == 3:
            if line[0] == '0':
                n += 1
                new_file_neg.writelines(line[1] + '\001' + line[2] + '\n')
            elif line[0] == '1':
                p += 1
                new_file_pos.writelines(line[1] + '\001' + line[2] + '\n')
            else:
                j += 1
    print "n=", n, "p=", p

    print j
    new_file_pos.close()
    new_file_neg.close()

def jieba_cut(filename,seg_filename):
    start = time.clock()
    jieba.suggest_freq('喵喵',True)
    jieba.suggest_freq('羞嗒嗒',True)
    jieba.suggest_freq('挖鼻屎',True)
    jieba.suggest_freq('萌萌哒', True)
    jieba.suggest_freq('坟头草', True)
    jieba.suggest_freq('萌宠', True)
    jieba.suggest_freq('笑尿了 ', True)
    jieba.suggest_freq('什么鬼', True)
    jieba.suggest_freq('小清新', True)
    jieba.suggest_freq('赞你', True)
    jieba.suggest_freq('正经汪', True)
    jieba.suggest_freq('笑尿', True)
    jieba.suggest_freq('污力', True)
    jieba.suggest_freq('小三', True)
    jieba.suggest_freq('狗日的', True)
    jieba.suggest_freq('心累', True)
    jieba.suggest_freq('你娘', True)
    jieba.suggest_freq('你妈', True)
    jieba.suggest_freq('狗日的', True)
    jieba.suggest_freq('蒙蔽了', True)
    jieba.suggest_freq('懵逼', True)
    jieba.suggest_freq('画风', True)
    jieba.suggest_freq('爆料', True)
    jieba.suggest_freq('大数据', True)
    jieba.suggest_freq('虐心', True)
    jieba.suggest_freq('虐哭', True)
    jieba.suggest_freq('单身汪', True)
    jieba.suggest_freq('单身狗', True)
    jieba.suggest_freq('真实伤害', True)
    jieba.suggest_freq('狗粮', True)
    jieba.suggest_freq('美男子', True)
    jieba.suggest_freq('萌死', True)
    jieba.suggest_freq('妈的智障', True)
    jieba.suggest_freq('深井冰', True)
    jieba.suggest_freq('蛇精病', True)
    jieba.suggest_freq('的日常', True)
    jieba.suggest_freq('女神经', True)
    jieba.suggest_freq('有颜', True)
    jieba.suggest_freq('嘿嘿嘿', True)
    jieba.suggest_freq('校园暴力', True)
    jieba.suggest_freq('心好累', True)
    jieba.suggest_freq('笑死', True)
    jieba.suggest_freq('激萌', True)
    jieba.suggest_freq('没天理', True)
    jieba.suggest_freq('小船', True)
    jieba.suggest_freq('说翻就翻', True)
    jieba.suggest_freq('不忍直视', True)
    jieba.suggest_freq('撩妹', True)
    jieba.suggest_freq('哭死了', True)
    jieba.suggest_freq('撕逼', True)
    jieba.suggest_freq('丑萌', True)
    jieba.suggest_freq('暴露年龄', True)
    jieba.suggest_freq('牛逼', True)
    jieba.suggest_freq('嗨皮', True)
    jieba.suggest_freq('好湿', True)
    jieba.suggest_freq('撩人', True)
    jieba.suggest_freq('翻牌', True)
    jieba.suggest_freq('逗比', True)
    jieba.suggest_freq('泪牛满面', True)
    jieba.suggest_freq('日他', True)
    jieba.suggest_freq('日你妈', True)
    jieba.suggest_freq('妈屁', True)
    jieba.suggest_freq('老母', True)
    jieba.suggest_freq('日他娘', True)
    jieba.suggest_freq('日本狗', True)
    jieba.suggest_freq('老子', True)
    jieba.suggest_freq('操你妈', True)
    jieba.suggest_freq('操你', True)
    jieba.suggest_freq('日本狗', True)
    jieba.suggest_freq('有毛', True)
    jieba.suggest_freq('傻逼', True)
    jieba.suggest_freq('有毛', True)
    jieba.suggest_freq('缺德', True)
    jieba.suggest_freq('艹死', True)
    jieba.suggest_freq('真他妈', True)
    jieba.suggest_freq('真他妈', True)
    jieba.suggest_freq('真他妈', True)
    new_file_pos_seg= file(seg_filename, 'w')
    nfile = open(filename,'r')
    seg_doc = []
    for line in nfile.readlines():
        seg_list = jieba.cut(line, cut_all=False)

        # print type(seg_list)
        seg_doc.append(" ".join(seg_list))
    print seg_doc
    for doc in seg_doc:
        doc_str = doc.encode("utf-8")
        new_file_pos_seg.writelines(doc_str)
    new_file_pos_seg.close()
    end = time.clock()
    print "consume time: %f s" % (end - start)

def filter_symbol():
    new_file = file('/home/houzhuo1994cs/Documents/lda/neg_no_punc.txt','w')
    oldfile = open('/home/houzhuo1994cs/Documents/lda/neg_seg.txt', 'r')
    # pattern = re.compile("[\u4e00-\u9fa5]+")
    for line in oldfile.readlines():
        line = unicode(line, 'utf-8')
        m = re.findall(ur"[\u4e00-\u9fa5]+", line)
        print m
        if m:
            str1 = ' '.join(m)  # 同行的中文用竖杠区分
            str2 = str(str1)
            new_file.write(str2)  # 写入文件
        new_file.write("\n")  # 不同行的要换行

        # pattern = re.compile(r"(/)|(！)|(【)|(】)|()|(，)|(:)|(”)|(“)|(#)|(。)")

        # no_sysmbol = re.sub(pattern, "", line)
        # new_file.writelines(no_sysmbol+'\n')
    new_file.close()


def stop_word():
    new_file = file('/home/houzhuo1994cs/Documents/lda/pos_seg_nopunc_stop.txt', 'w')
    final = ''
    stop_dict = {}.fromkeys([ line.rstrip() for line in open('/home/houzhuo1994cs/Documents/lda/609004/1_utf8.txt') ])
    # print stop_dict
    final = ''
    read_file = codecs.open('/home/houzhuo1994cs/Documents/lda/pos_seg_nosymbol.txt', 'r', 'utf8')
    for words in read_file:

        for word in words:
            word = word.encode('utf8')

            if word not in stop_dict:
                final += word
    new_file.write(final)
    new_file.close()
    print final

def extract_sentiment_word():
    start = time.clock()
    new_file = file('/home/houzhuo1994cs/Documents/lda/neg_sentiment.txt', 'w')
    sentiment_dict = {}.fromkeys([line.rstrip() for line in open('/home/houzhuo1994cs/Documents/lda/dic/tsinghua.negative.utf8.txt')])
    #final = ''
    # read_file = codecs.open('/home/houzhuo1994cs/Documents/lda/pos_seg_nopunc_stop.txt', 'r', 'utf8')
    for line in open('/home/houzhuo1994cs/Documents/lda/neg_no_punc.txt', 'r'):
        line = line.strip('\n').split(' ')
        for word in line:
            word = word.encode('utf-8')
            if len(word) > 3:
                if word in sentiment_dict:
                    new_file.write(word + " ")
        new_file.write("\n")
    new_file.close()
    end = time.clock()
    print "read: %f s" % (end - start)


def add_line_number_orLabel():
    n = 0
    start = time.clock()
    new_file = file('/home/houzhuo1994cs/Documents/lda/neg_sentiment_no_label.txt', 'w')
    for line in open('/home/houzhuo1994cs/Documents/lda/neg_sentiment.txt', 'r'):
        line = line.strip('\n').split(' ')
        if len(line) != 1:
            print len(line)
            #n += 1
            #new_file.write(str(n)+' ')
            for word in line:
                new_file.write(word + " ")
            #new_file.write('1')
            new_file.write('\n')
    new_file.close()


def filter_sim():

    dict = {}
    content = {}

    new_file = file('/home/houzhuo1994cs/Documents/lda/pos_nosim.txt', 'w')
    for line in open('/home/houzhuo1994cs/Documents/lda/pos_sentiment_line.txt', 'r'):
        line = line.strip('\n').split(' ')
        num = int(line[0])
        print len(line)
        dict.setdefault(num,len(line))
        content.setdefault(num,line[1:])
    print num
    for n in xrange(num):
        if dict.get(n) > 3:
            #print dict.get(n),dict.get(n+1)
            if content.get(n):
                if dict.get(n) != dict.get(n + 1):
                    for word in content.get(n):
                        new_file.write(word + " ")
        new_file.write("1")
        new_file.write("\n")
    new_file.close()




def btmIndex(filename):
    new_file = file('/home/houzhuo1994cs/Documents/BTM-master/output/pos_data.txt', 'w')
    dict = {}
    for line in open('/home/houzhuo1994cs/Documents/BTM-master/output/voca.txt','r'):
        lineArr = line.strip('\n').split()
        dict.setdefault(lineArr[1],lineArr[0])
    print dict.get('问题')
    for line2 in open(filename,'r'):
        lineArr2 = line2.strip('\n').split()
        for word in lineArr2:
            if dict.get(word):
                new_file.write(dict.get(word)+ ' ')
        new_file.write('\n')
    new_file.close()

def showTopic():
    new_file = file('/home/houzhuo1994cs/Documents/BTM-master/output/pos_topic.txt', 'w')
    dict = {}
    for line in open('/home/houzhuo1994cs/Documents/BTM-master/output/model/k10.pz_d','r'):
        lineArr = line.strip('\n').split()
        for index,number in enumerate(lineArr):
            num = "%.5f" % float(number)
            dict.setdefault(index,num)
        t1 = sorted(dict.iteritems(),key = lambda d:d[1],reverse = True )[0][0]
        t2 = sorted(dict.iteritems(),key = lambda d:d[1],reverse = True )[1][0]
        new_file.write(str(t1)+' ')
        new_file.write(str(t2)+' ')
        new_file.write('\n')
        dict.clear()
    new_file.close()


# def bulit_vsm():
#     new_file = file('/home/houzhuo1994cs/Documents/BTM-master/output/svm_train.txt', 'w')
#     n = 0
#     dict = {}
#     for line in open('/home/houzhuo1994cs/PycharmProjects/LDA/btm/voca.txt', 'r'):
#         lineArr = line.strip('\n').split()
#         dict.setdefault(lineArr[1], lineArr[0])
#     for line in open('/home/houzhuo1994cs/Documents/BTM-master/output/data.txt', 'r'):







if __name__ == '__main__':
    # filename = '/home/houzhuo1994cs/Documents/lda/weibo_base_filter_neg (1).txt'
    # seg_filename = '/home/houzhuo1994cs/Documents/lda/neg_seg.txt'
    # jieba_cut(filename,seg_filename)

    # filter_symbol()

    #stop_word()

    #extract_sentiment_word()
    #add_line_number_orLabel()
    # filter_sim()

    #btmIndex('/home/houzhuo1994cs/Documents/lda/pos_sentiment_no_label.txt')
   # filter_sim()
   showTopic()