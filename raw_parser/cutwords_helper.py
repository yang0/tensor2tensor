#coding=utf-8
from __future__ import print_function
# import jieba
# import os
#
# _ROOTDIR = os.path.dirname(os.path.realpath(__file__))
# STOP_WORDS = _ROOTDIR+"/jieba_dict/stopwords.txt"
#
# MEDICINE_DICT = _ROOTDIR+"/jieba_dict/medicine.dict"
# USER_DICT = _ROOTDIR+"/jieba_dict/user.dict"
# DICT_LIST = [MEDICINE_DICT, USER_DICT]
#
# stopwordset = set()
#
# def loadUserDicts():
#     for dict in DICT_LIST:
#         jieba.load_userdict(dict)


#定义停词
# def loadStopWords():
#     with open(STOP_WORDS, 'r', encoding='utf-8') as sw:
#         for line in sw:
#             stopwordset.add(line.strip('\n'))



def cutWords(s):
    # words = jieba.cut(s, cut_all=False)
    words = s

    l = list(words)
    length = len(l)
    if length > 150:
        print("句子超长, %d个词，忽略" % length)
        return ""

    return " ".join(l)


# def cutWords(s, stop=False, ignoreLongSentence=True):
#     # words = jieba.cut(s, cut_all=False)
#     words = s
#
#     l = list(words)
#     length = len(l)
#     if length > 150:
#         print("句子超长, %d个词，忽略" % length)
#         return ""
#
#     return " ".join(l)

    # if stop == False:
    #     return " ".join(l)

    # rs = ''
    # for word in l:
    #     if word not in stopwordset:
    #         rs += word + ' '

    # return rs

# #将多组会话分词
# def cutConversations(conversations):
#     newConversations = []
#     for conversation in conversations:
#         #print(conversation[1])
#         newConversation = [cutWords(conversation[0]).strip(), "\n", cutWords(conversation[1]).strip(), "\n", "E", "\n"]
#         if newConversation[0] != "" and newConversation[2] != "":
#             newConversations += newConversation
#
#     return newConversations
#
#
#
#
# #加载自定义词典
# loadUserDicts()
# loadStopWords()
