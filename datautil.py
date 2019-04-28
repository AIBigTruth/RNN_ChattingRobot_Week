import sys
import os
import matplotlib.mlab as mlab
import matplotlib.pyplot as plt
import numpy as np
from tensorflow.python.platform import gfile
from random import shuffle
import re
import collections
import jieba

jieba.load_userdict("myjiebadict.txt")

"""------------------------------------------------
filesfrom,_=getRawFileList(data_dir+"fromids/")
filesto,_=getRawFileList(data_dir+"toids/")
data_dir = "datacn/"
raw_data_dir = "datacn/dialog/"
raw_data_dir_to = "datacn/dialog/"
vocabulary_fileen ="dicten.txt"
vocabulary_filech = "dictch.txt"
-------------------------------------------------"""

"""函数1：获取文件列表"""
def getRawFileList(path):
    """---------------------------------------------------------------
    函数功能：获取数据文件的列表
    函数应用：（5）获取文件文本：get_ch_path_text(raw_data_dir,Isch=True,normalize_digits=False)
    函数输入：path: raw_data_dir          "datacn/dialog"
    函数输出：files: ['datacn/dialog/one.txt', 'datacn/dialog/two.txt']
             names: ['one.txt', 'two.txt']
    -----------------------------------------------------------------"""
    files = []
    names = []
    for f in os.listdir(path):  # 返回指定的文件夹包含的文件或文件夹的名字的列表
        if not f.endswith("~") or not f == "":
            files.append(os.path.join(path, f))  # 用于路径拼接文件路径
            names.append(f)  # 文件名
    return files, names


"""函数2：分词函数"""
def fenci(training_data):
    """-------------------------------------------------------
    函数功能：将数据过滤后的纯文字进行分词（去标点符号、去数字等等过滤）
    函数应用：fenci(notoken);(4）从文件（one.txt，two.txt）中读取中文词：get_ch_lable(txt_file,Isch=True,normalize_digits=False)
    函数输入：seg_list = jieba.cut("他来到上海交通大学", cut_all=True)
              print("【全模式】：" + "/ ".join(seg_list))
    函数输出：他/ 来到/ 上海交通大学
    ----------------------------------------------------------"""
    seg_list = jieba.cut(training_data)  # 默认是精确模式
    training_ci = " ".join(seg_list)
    training_ci = training_ci.split()
    return training_ci


"""函数3：基本记录器"""
def basic_tokenizer(sentence):
    """---------------------------------------------------
    函数功能：把语聊中有，。等地方去掉符号，直接相接
    函数应用：(4）从文件（one.txt，two.txt）中读取中文词：get_ch_lable(txt_file,Isch=True,normalize_digits=False)
    函数输入：basic_tokenizer(linstr1);sentence=""hsdhd，2./8d；xd！s}。p:dss""
    函数输出：str1： hsdhd2./8d；xd！s}p:dss
                 str2:  hsdhd2./8d；xd！s}p:dss
    -----------------------------------------------------"""
    _WORD_SPLIT = "([.,!?\"':;)(])"  # _word_split
    _CHWORD_SPLIT = '、|。|，|‘|’'  # _chword_split
    str1 = " "
    for i in re.split(_CHWORD_SPLIT, sentence):  # 使用re.split()方法，可以为分隔符指定多个模式(_CHWORD_SPLIT)里面的模式
        str1 = str1 + i
    str2 = " "
    for i in re.split(_WORD_SPLIT, str1):
        str2 = str2 + i
    return str2


"""--------------------------------------------------
系统字符，创建字典时需要加入，特殊标记，用来填充标记对话
如"i love you""i love you" 创建成vocabvocab 时，
应为："_GO i love you _EOS
-----------------------------------------------------"""
_NUM = "_NUM"  # 文字字符替换，不属于系统字符
_PAD = "_PAD"  # 空白补位
_GO = "_GO"  # 对话开始
_EOS = "_EOS"  # 对话结束
_UNK = "_UNK"  # 标记未出现在词汇表中的字符
PAD_ID = 0  # 特殊标记对应的向量值
GO_ID = 1  # 特殊标记对应的向量值
EOS_ID = 2  # 特殊标记对应的向量值
UNK_ID = 3  # 特殊标记对应的向量值

"""函数4：从文件（one.txt，two.txt）中读取中文词，lable->word"""
def get_ch_lable(txt_file, Isch=True, normalize_digits=False):
    """-------------------------------------------------------
    函数功能：数据预处理，去符号，去数字，分词
    函数接收：（2）分词：fenci(notoken)
             （3）基本记录器：basic_tokenizer(linstr1)
    函数应用：（5）获取文件中的文本：get_ch_path_text(raw_data_dir,Isch=True,normalize_digits=False):
    函数输入：txt_file: one.txt,two.txt里面的文字
    函数输出：lable:['到'  '了' '吗' '到' '了' '啊',,,,,]
             lablessz:[0, 0, 0, 3, 8, 12, 15, 19, 19, 19, 21,,,,569]
    isch:代表是否是按中文方式处理；normalize_digits：代表是否将数字替换掉
    (1)list() 方法用于将元组转换为列表。
    (2)decode的作用是将其他编码的字符串转换成unicode编码，如label.decode('gb2312')，
    表示将gb2312编码的字符串转换成unicode编码。
    (3)str = "Line1-abcdef \nLine2-abc \nLine4-abcd";
       print str.split( );       # 以空格为分隔符，包含 \n
      ['Line1-abcdef', 'Line2-abc', 'Line4-abcd']
    (4) txt_file:你多大了,你猜猜
    (5) label->word:你。。。。多。。。。大
        labelssz:13689....不断变大
    ----------------------------------------------------------"""
    labels = list()  # 装一个一个切片好的中文词，切好一个扔进去一个，重复
    labelssz = []  # len(labels)，越来越大
    with open(txt_file, 'rb') as f:
        for label in f:
            # linstr1 =label.decode('utf-8')
            linstr1 = label.decode('gbk')
            #linstr1 = label.decode('gb2312')
            if normalize_digits:
                linstr1 = re.sub('\d+', _NUM, linstr1)  # 正则表达式，匹配任意的数字，并将其替换（去数字）
            notoken = basic_tokenizer(linstr1)  # （去标点符号，。等等），->纯文字后再进行分词
            if Isch:
                notoken = fenci(notoken)
            else:
                notoken = notoken.split()  # 通过指定分隔符对字符串进行切片
            labels.extend(notoken)  # 在列表末尾一次性追加另一个序列中的多个值
            labelssz.append(len(labels))  # 在列表末尾添加新的对象
    return labels, labelssz  # 返回切好的词，和，词大小的长度信息


""" 函数5：获取路径文件中的文本 """
def get_ch_path_text(raw_data_dir, Isch=True, normalize_digits=False):
    """---------------------------------------------
    函数接收：（1）获取文件列表:getRawFileList(raw_data_dir)
             （4）从文件中读取中文词:get_ch_lable(text_file, Isch, normalize_digits)
    函数输入：raw_data_dir = "datacn/dialog/"   one.txt,two.txt
    函数输出：labels:['到'  '了' '吗' '到' '了' '啊',,,,,]
             training_dataszs:[0, 0, 0, 3, 8, 12, 15, 19, 19, 19, 21,,,,569]
    ---------------------------------------------------"""
    # text_files: ['datacn/dialog/one.txt', 'datacn/dialog/two.txt']
    text_files, _ = getRawFileList(raw_data_dir)
    labels = []
    training_dataszs = list([0])  # [0]
    if len(text_files) == 0:
        print("err:no files in ", raw_data_dir)
        return labels
    print("len(text_files):", len(text_files), ";files,one is", text_files[0], ";files,twe is", text_files[1])
    """------------------------------------------------
     shuffle() 方法将序列的所有元素随机排序
     list = [20, 16, 10, 5];random.shuffle(list)
     随机排序列表 :  [16, 5, 10, 20]
     ---------------------------------------------------"""
    shuffle(text_files)

    for text_file in text_files:
        """---------------------------------------------
           one.txt in text_files;two.txt in text_files
           get_ch_lable()->return labels, labelssz
           training_data:['到'  '了' '吗' '到' '了' '啊',,,,,]
           training_datasz:[0, 0, 0, 3, 8, 12, 15, 19, 19, 19, 21,,,,569]
           -----------------------------------------------"""
        training_data, training_datasz = get_ch_lable(text_file, Isch, normalize_digits)
        training_ci = np.array(training_data)  # 变数组
        training_ci = np.reshape(training_ci, [-1, ])  # 一维
        labels.append(training_ci)  # 放list里面
        training_datasz = np.array(training_datasz) + training_dataszs[-1]
        training_dataszs.extend(list(training_datasz))
        print("training_dataszs:",
              training_dataszs)  # training_dataszs: [0, 4, 6, 11, 14, 17,,,,200];training_dataszs: [0, 4, 6, 11, 14, 17, 21,,569]
    return labels, training_dataszs


"""函数6：建立数据集"""
def build_dataset(words, n_words):
    """------------------------------------------------
    函数功能：将原始输入处理得到数据集中
    函数应用：
    函数输入：words
    函数输出：data:[4, 18, 30, 7, 4, 91,,,,]
             count:
             dictionary:
             reversed_dictionary:{0: '_PAD', 1: '_GO', 2: '_EOS', 3: '_UNK', 4: '你', 5: '我', ,,,239:}
    most_common()函数用来实现Top n 功能.
    user_counter = Counter("abbafafpskaag")
    print(user_counter.most_common(3)) #[('a', 5), ('b', 2), ('f', 2)]"""
    count = [[_PAD, -1], [_GO, -1], [_EOS, -1], [_UNK, -1]]
    count.extend(collections.Counter(words).most_common(n_words - 1))
    dictionary = dict()  # 创建空字典
    for word, _ in count:
        dictionary[word] = len(dictionary)  # 0: '_PAD', 1: '_GO', 2: '_EOS', 3: '_UNK'，123345就是有字典长度得到
    data = list()  # 创建空list
    unk_count = 0
    for word in words:
        if word in dictionary:
            index = dictionary[word]  # 词->指针（数字）
        else:
            index = 0  # dictionary['UNK']
            unk_count += 1  # 统计不在字典里面的字有多少个
        data.append(index)  # 将数字放入data中
    count[0][1] = unk_count
    # zip() 函数用于将可迭代的对象作为参数，将对象中对应的元素打包成一个个元组，然后返回由这些元组组成的对象
    # 再将元组dict成字典类型
    reversed_dictionary = dict(zip(dictionary.values(), dictionary.keys()))
    return data, count, dictionary, reversed_dictionary


""" ---------------------------------------------------------------
函数7：创建词典
Isch=true 中文， false 英文，创建词典 max_vocabulary_size=500 500个词 
将file 中的字符出现的次数进行统计，并按照从小到大的顺序排列，每个字符对应
的排序序号就是它在词典中的编码，这样就形成了一个 key-vlaue 的字典查询表。
不管原训练语料是什么语言都没有区别了，因为对于训练模型来说都是数字化的向量。
----------------------------------------------------------------------"""
def create_vocabulary(vocabulary_file, raw_data_dir, max_vocabulary_size, Isch=True, normalize_digits=True):
    """-------------------------------------------------------------------
    函数功能：从路径中的文件里创建字典
    函数接收：（5）从路径中获取文件中的文本：get_ch_path_text（raw_data_dir）
              (6) 建立数据集：build_dataset(all_words,max_vocabulary_size)
    函数应用：
    函数输入：
    函数输出：
    ----------------------------------------------------------------------"""
    texts, textssz = get_ch_path_text(raw_data_dir, Isch, normalize_digits)  # return words, length
    print("texts[0]:", texts[0], "len(texts0):", len(texts[0]))  # texts[0]: ['到' '了' '吗' '到',,,,,];len(texts): 2
    print("texts[1]:", texts[1], "len(texts1):", len(texts[1]))
    print("行数:", len(textssz), "textssz:", textssz)  # 行数: 249 ;textssz:  [0, 0, 0, 3, 8, 12, 15, ,,,,569]
    all_words = []
    for label in texts:
        print("词数:", len(label))  # 词数: 3059,3160
        all_words += [word for word in label]
    print("总词数:", len(all_words))  # 词数: 6219

    training_label, count, dictionary, reverse_dictionary = build_dataset(all_words, max_vocabulary_size)
    print("reverse_dictionary:", reverse_dictionary, len(reverse_dictionary))
    # reverse_dictionary {0: '_PAD', 1: '_GO', 2: '_EOS', 3: '_UNK', 4: '你', 5: '我', 6: '的',,,,239:'那个]'
    if not gfile.Exists(vocabulary_file):
        print("Creating vocabulary %s from data %s" % (vocabulary_file, data_dir))
        if len(reverse_dictionary) > max_vocabulary_size:
            reverse_dictionary = reverse_dictionary[:max_vocabulary_size]
        with gfile.GFile(vocabulary_file, mode="w") as vocab_file:
            for w in reverse_dictionary:
                print("reverse_dictionary[w]", reverse_dictionary[w])
                vocab_file.write(reverse_dictionary[w] + "\n")
    else:
        print("already have vocabulary!  do nothing !!!!!!!!!!!!!!!!!!!!!!!!!!!!!")
    return training_label, count, dictionary, reverse_dictionary, textssz


"""函数8：把data中的内存问和答的ids数据放在不同的文件里 """
def create_seq2seqfile(data, sorcefile, targetfile, textssz):
    """-------------------------------------------------------------
    函数功能：把问的id和答的ID分开来放
    函数接收：
    函数应用：（9）：create_seq2seqfile(training_data, source_file , target_file, textssz)
    函数输入：data
    函数输出：data_source_test.txt, data_target_test.txt
    -------------------------------------------------------------------"""
    print("data:", data, "len(data):", len(data))
    # data: [33, 7, 15, 33, 7, 12, 19, 8, ,,,]; len(data): 569=200+369
    with open(sorcefile, 'w') as sor_f:
        with open(targetfile, 'w') as tar_f:
            for i in range(len(textssz) - 1):
                print("textssz", i, textssz[i], textssz[i + 1], data[textssz[i]:textssz[i + 1]])
                """ """
                if (i + 1) % 2:
                    sor_f.write(str(data[textssz[i]:textssz[i + 1]]).replace(',', ' ')[1:-1] + '\n')
                else:
                    tar_f.write(str(data[textssz[i]:textssz[i + 1]]).replace(',', ' ')[1:-1] + '\n')


"""函数9：将读好的对话文本按行分开，一行问，一行答。存为两个文件。training_data为总数据，textssz为每行的索引"""
def splitFileOneline(training_data, textssz):
    source_file = os.path.join(data_dir + 'fromids/', "data_source_test.txt")
    target_file = os.path.join(data_dir + 'toids/', "data_target_test.txt")
    create_seq2seqfile(training_data, source_file, target_file, textssz)
    return source_file, target_file


"""函数10：将句子转成ids"""
def sentence_to_ids(sentence, vocabulary, normalize_digits=True, Isch=True):
    if normalize_digits:
        sentence = re.sub('\d+', _NUM, sentence)
    notoken = basic_tokenizer(sentence)
    if Isch:
        notoken = fenci(notoken)
    else:
        notoken = notoken.split()

    idsdata = [vocabulary.get(w, UNK_ID) for w in notoken]
    return idsdata


"""函数11：将一个文件转成ids 不是windows下的要改编码格式 utf8"""
def textfile_to_idsfile(data_file_name, target_file_name, vocab, normalize_digits=True, Isch=True):
    if not gfile.Exists(target_file_name):
        print("Tokenizing data in %s" % data_file_name)
        with gfile.GFile(data_file_name, mode="rb") as data_file:
            with gfile.GFile(target_file_name, mode="w") as ids_file:
                counter = 0
                for line in data_file:
                    counter += 1
                    if counter % 100000 == 0:
                        print("  tokenizing line %d" % counter)
                    token_ids = sentence_to_ids(line.decode('gb2312'), vocab, normalize_digits, Isch)

                    ids_file.write(" ".join([str(tok) for tok in token_ids]) + "\n")


"""函数12：将文件批量转成ids文件"""
def textdir_to_idsdir(textdir, idsdir, vocab, normalize_digits=True, Isch=True):
    text_files, filenames = getRawFileList(textdir)

    if len(text_files) == 0:
        raise ValueError("err:no files in ", raw_data_dir_to)

    print(len(text_files), "files,one is", text_files[0])

    for text_file, name in zip(text_files, filenames):
        print(text_file, idsdir + name)
        textfile_to_idsfile(text_file, idsdir + name, vocab, normalize_digits, Isch)


def ids2texts(indices, rev_vocab):
    texts = []
    for index in indices:
        texts.append(rev_vocab[index])
    return texts

def initialize_vocabulary(vocabulary_path):
  if gfile.Exists(vocabulary_path):
    rev_vocab = []
    with gfile.GFile(vocabulary_path, mode="r") as f:
      rev_vocab.extend(f.readlines())
    rev_vocab = [line.strip() for line in rev_vocab]
    vocab = dict([(x, y) for (y, x) in enumerate(rev_vocab)])
    return vocab, rev_vocab
  else:
    raise ValueError("Vocabulary file %s not found.", vocabulary_path)

"""绘制条形图"""
def plot_scatter_lengths(title, x_title, y_title, x_lengths, y_lengths):
    plt.scatter(x_lengths, y_lengths)
    plt.title(title)
    plt.xlabel(x_title)
    plt.ylabel(y_title)
    plt.ylim(0, max(y_lengths))
    plt.xlim(0, max(x_lengths))
    plt.show()


def plot_histo_lengths(title, lengths):
    mu = np.std(lengths)
    sigma = np.mean(lengths)
    x = np.array(lengths)
    n, bins, patches = plt.hist(x, 50, facecolor='green', alpha=0.5)
    y = mlab.normpdf(bins, mu, sigma)
    plt.plot(bins, y, 'r--')
    plt.title(title)
    plt.xlabel("Length")
    plt.ylabel("Number of Sequences")
    plt.xlim(0, max(lengths))
    plt.show()


"""分析文本"""
def analysisfile(source_file, target_file):
    source_lengths = []
    target_lengths = []
    with gfile.GFile(source_file, mode="r") as s_file:
        with gfile.GFile(target_file,
                         mode="r") as t_file:  # 获取文本操作句柄，类似于python提供的文本操作open()函数，filename是要打开的文件名，mode是以何种方式去读写，将会返回一个文本操作句柄。
            source = s_file.readline()
            target = t_file.readline()
            counter = 0
            while source and target:
                counter += 1
                if counter % 100000 == 0:
                    print("  reading data line %d" % counter)
                    sys.stdout.flush()
                num_source_ids = len(source.split())
                source_lengths.append(num_source_ids)
                num_target_ids = len(target.split()) + 1  # plus 1 for EOS token
                target_lengths.append(num_target_ids)
                source, target = s_file.readline(), t_file.readline()
    print("target_lengths:", target_lengths, "source_lengths:", source_lengths)
    if plot_histograms:
        plot_histo_lengths("target lengths", target_lengths)
        plot_histo_lengths("source_lengths", source_lengths)
    if plot_scatter:
        plot_scatter_lengths("target vs source length", "source length", "target length", source_lengths,
                             target_lengths)


data_dir = "datacn/"
raw_data_dir_to = "datacn/dialog/"
vocabulary_filech = "dictch.txt"
plot_histograms = plot_scatter = True
vocab_size = 40000
max_num_lines = 1
max_target_size = 200
max_source_size = 200


def main():
    vocabulary_filenamech = os.path.join(data_dir, vocabulary_filech)  # 生成的字典文件路径data_dir+vocabulary_filech
    """创建中文字典 """
    training_datach, countch, dictionarych, reverse_dictionarych, textsszch \
        = create_vocabulary(vocabulary_filenamech, raw_data_dir_to, vocab_size, Isch=True, normalize_digits=True)
    source_file, target_file = splitFileOneline(training_datach, textsszch)
    print("training_datach:", len(training_datach))
    print("dictionarych:", len(dictionarych))
    analysisfile(source_file, target_file)


#    #textdir_to_idsdir(raw_data_dir,data_dir+"fromids/",vocaben,normalize_digits=True,Isch=False)
#    textdir_to_idsdir(raw_data_dir_to,data_dir+"toids/",vocabch,normalize_digits=True,Isch=True)
#    filesfrom,_=getRawFileList(data_dir+"fromids/")
#    filesto,_=getRawFileList(data_dir+"toids/")
#    source_train_file_path = filesfrom[0]
#    target_train_file_path= filesto[0]
#    analysisfile(source_train_file_path,target_train_file_path)


if __name__ == "__main__":
    main()