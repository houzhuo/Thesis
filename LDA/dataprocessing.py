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

#def stop_word():
#     new_file = file('/home/houzhuo1994cs/Documents/lda/pos_seg_nopunc_stop.txt', 'w')
#     final = ''
#     stop_dict = {}.fromkeys([ line.rstrip() for line in open('/home/houzhuo1994cs/Documents/lda/609004/1_utf8.txt') ])
#     # print stop_dict
#     final = ''
#     read_file = codecs.open('/home/houzhuo1994cs/Documents/lda/pos_seg_nosymbol.txt', 'r', 'utf8')
#     for words in read_file:
#
#         for word in words:
#             word = word.encode('utf8')
#
#             if word not in stop_dict:
#                 final += word
#     new_file.write(final)
#     new_file.close()
#     print final
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
    new_file = file('C:\Users\Zhuo\Documents\weibo_base_filter_pos.txt', 'w')
    for line in open('C:\Users\Zhuo\Documents\pos_weibo_base_new.txt', 'r'):
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


def filter_sentiment_dict():
    new_file = file('/home/houzhuo1994cs/Documents/lda/dic/sentiment_dict_new.txt','w')
    fr = open('/home/houzhuo1994cs/Documents/lda/dic/sentiment_dict.txt','r')
    for line in fr.readlines():
        lineArr = line.strip('\n').split()
        for word in lineArr:
            word = word.encode('utf8')
            if len(word)>3:
                new_file.write(word+' ')
        new_file.write('\n')
    new_file.close()
    fr.close()


def jieba_cut(filename):
    start = time.clock()
    print "=========== cut word ==========="
    fr = open('/home/houzhuo1994cs/Documents/lda/dic/sentiment_dict_new.txt')
    for line in fr.readlines():
        lineArr = line.strip('\n').split()
        for word in lineArr:
            jieba.suggest_freq(word, True)

    new_file_seg= file('/home/houzhuo1994cs/Documents/lda/feature/neg_seg.txt', 'w')
    nfile = open(filename,'r')
    seg_doc = []
    for line in nfile.readlines():
        seg_list = jieba.cut(line, cut_all=False)

        # print type(seg_list)
        seg_doc.append(" ".join(seg_list))
    print "==========seg finish ============="
    for doc in seg_doc:
        doc_str = doc.encode("utf-8")
        new_file_seg.writelines(doc_str)
    end = time.clock()
    print "consume time: %f s" % (end - start)
    new_file_seg.close()

    print "==========filter_symbol=========="

    new_file_seg_for_extrc = open('/home/houzhuo1994cs/Documents/lda/feature/neg_seg.txt')
    no_punc_filename = '/home/houzhuo1994cs/Documents/lda/feature/neg_no_punc.txt'


    new_file = file(no_punc_filename,'w')
    # pattern = re.compile("[\u4e00-\u9fa5]+")
    for line in new_file_seg_for_extrc.readlines():
        line = unicode(line, 'utf-8')
        m = re.findall(ur"[\u4e00-\u9fa5]+", line)
        if m:
            str1 = ' '.join(m)  # 同行的中文用竖杠区分
            str2 = str(str1)
            new_file.write(str2)  # 写入文件
        new_file.write("\n")  # 不同行的要换行
    new_file_seg_for_extrc.close()
    new_file.close()
    nfile.close()
    fr.close()
    return no_punc_filename


def extract_sentiment_word():

    start = time.clock()
    no_punc_file_name = jieba_cut(filename)
    #no_punc_file_name = '/home/houzhuo1994cs/Documents/lda/feature/base_no_punc.txt'
    print '==========extract sentiment========='
    base_sentiment_file_name = '/home/houzhuo1994cs/Documents/lda/feature/neg_sentiment.txt'
    new_file = file(base_sentiment_file_name, 'w')
    sentiment_dict = {}.fromkeys([line.rstrip() for line in open('/home/houzhuo1994cs/Documents/lda/dic/sentiment_dict_new.txt')])
    for line in open(no_punc_file_name, 'r'):
        line = line.strip('\n').split(' ')
        for word in line:
            word = word.encode('utf-8')
            #if len(word) > 3:
            if word in sentiment_dict:
                new_file.write(word + " ")
        new_file.write("\n")
    new_file.close()
    end = time.clock()
    print "read: %f s" % (end - start)
    return base_sentiment_file_name


def add_line_number_orLabel():

    n = 0
    start = time.clock()
    base_sentiment_file_name = extract_sentiment_word()
    print '==========add line ========='
    base_sentiment_line = '/home/houzhuo1994cs/Documents/lda/feature/neg_sentiment_line.txt'
    new_file = file(base_sentiment_line, 'w')
    for line in open(base_sentiment_file_name, 'r'):
        line = line.strip('\n').split(' ')
        if len(line) != 1:
            print len(line)
            n += 1
            new_file.write(str(n)+' ')
            for word in line:
                new_file.write(word + " ")
            #new_file.write('1')
            new_file.write('\n')
    new_file.close()
    return base_sentiment_line


def filter_sim():

    dict = {}
    content = {}
    base_sentiment_line = add_line_number_orLabel()
    print '==========filter sim========='
    sentiment_nosim_name = '/home/houzhuo1994cs/Documents/lda/feature/neg_sentiment_nosim.txt'
    new_file = file(sentiment_nosim_name, 'w')
    for line in open(base_sentiment_line, 'r'):
        line = line.strip('\n').split(' ')
        num = int(line[0])
        # print len(line)
        dict.setdefault(num,len(line))
        content.setdefault(num,line[1:])
    for n in xrange(num):
        if dict.get(n) > 3:
            #print dict.get(n),dict.get(n+1)
            if content.get(n):
                if dict.get(n) != dict.get(n + 1):
                    for word in content.get(n):
                        new_file.write(word + " ")
        #new_file.write("1")
        new_file.write("\n")
    new_file.close()

    return sentiment_nosim_name

def delet_blankline():
    filename = filter_sim()
    print '========delet blankline=========='
    #filename = '/home/houzhuo1994cs/Documents/lda/feature/sentiment_nosim.txt'
    new_filename = '/home/houzhuo1994cs/Documents/lda/feature/neg_sentiment_noline.txt'
    new_file = file(new_filename, 'w')
    fr = open(filename)
    for line in fr.readlines():
        lineArr = line.strip().split(' ')
        #print len(lineArr[0])
        if len(lineArr[0]) != 0:
            for word in lineArr:
                word = word.encode('utf8')
                new_file.write(word + ' ')
            new_file.write("\n")
    fr.close()
    new_file.close()
    return new_filename
#for corpus:
#1.delet_blankline()
#2.copy to doc_info
#3.run sh --> voca.txt

##for neg or pos:(don not forget add -1)
#1.delet_blankline()
#2.btmIndex()
#3.run single sh commond
#4.topicshow(),generate()

def btmIndex():
    filename = '/home/houzhuo1994cs/Documents/lda/feature/neg_sentiment_noline.txt'
    new_file = file('/home/houzhuo1994cs/Documents/BTM-master/output/neg_data.txt', 'w')
    dict = {}
    for line in open('/home/houzhuo1994cs/Documents/BTM-master/output/voca.txt','r'):
        lineArr = line.strip('\n').split()
        dict.setdefault(lineArr[1],lineArr[0])
    for line2 in open(filename,'r'):
        lineArr2 = line2.strip('\n').split()
        for word in lineArr2:
            if dict.get(word):
                new_file.write(dict.get(word)+ ' ')
        new_file.write('\n')
    new_file.close()

def showTopic():

    new_file = file('/home/houzhuo1994cs/Documents/BTM-master/output/neg_topic.txt', 'w')
    dict = {}
    for line in open('/home/houzhuo1994cs/Documents/BTM-master/output/model/k40.pz_d','r'):
        lineArr = line.strip('\n').split()
        for index,number in enumerate(lineArr):
            num = "%.5f" % float(number)
            dict.setdefault(index,num)
            # print num
        t1 = sorted(dict.iteritems(),key = lambda d:d[1],reverse = True )[0][0]
        # print t1
        # print sorted(dict.iteritems(),key = lambda d:d[1],reverse = True )
        t2 = sorted(dict.iteritems(),key = lambda d:d[1],reverse = True )[1][0]
        t1 =  t1/1.00000
        t2 =  t2/1.00000
        new_file.write(str(t1)+'')
        new_file.write(' ')
        new_file.write(str(t2)+'')
        new_file.write(' ')
        new_file.write('\n')
        dict.clear()
    new_file.close()
    print "=====show topic finish====="


def generate_Train_data():
    new_file = file('/home/houzhuo1994cs/Documents/BTM-master/output/neg_topic_label.txt', 'w')
    for line in open('/home/houzhuo1994cs/Documents/BTM-master/output/neg_topic.txt', 'r'):
        lineArr = line.strip('\n').split()
        new_file.write(lineArr[0]+',')
        new_file.write(lineArr[1]+',')
        new_file.write('-1')
        new_file.write('\n')
    new_file.close()







if __name__ == '__main__':
    filename = '/home/houzhuo1994cs/Documents/lda/weibo_base_filter_neg.txt'

    #delet_blankline()
    # btmIndex()

    showTopic()
    generate_Train_data()