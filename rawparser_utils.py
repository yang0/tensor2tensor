#coding=utf-8
from __future__ import print_function
from raw_parser.multi100w import Multi100w
from raw_parser.question import Question
from raw_parser.test import Test


#分词
_rawFiles={
    "test":Test,
     "multi100w": Multi100w,
    "diagnosis": Question
}


def main():
    print("分词：%s" % ', '.join(_rawFiles.keys()))
    for k, v in _rawFiles.items():
       v().splitTrainTestFile()


if __name__ == '__main__':
    main()
