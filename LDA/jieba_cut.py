# encoding=utf-8
import jieba

def jieba_cut(filename,seg_filename):
    jieba.suggest_freq('喵喵',True)
    jieba.suggest_freq('羞嗒嗒',True)
    new_file_pos_seg= file(seg_filename, 'w')
    nfile = open(filename,'r')
    seg_doc = []

    for line in nfile.readlines():
        seg_list = jieba.cut(line, cut_all=False)
        print type(seg_list)
        seg_doc.append(" ".join(seg_list))
    print seg_doc
    for doc in seg_doc:
        doc_str = doc.encode("utf-8")
        new_file_pos_seg.writelines(doc_str)
    new_file_pos_seg.close()






if __name__ == '__main__':
    filename = '/home/houzhuo1994cs/Documents/lda/weibo_base_filter_pos.txt'
    seg_filename = '/home/houzhuo1994cs/Documents/lda/weibo_filter_pos_seg.txt'
    jieba_cut(filename,seg_filename)