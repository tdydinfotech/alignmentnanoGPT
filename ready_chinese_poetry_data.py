import glob
import json
import os

"""
把所有唐诗的语句提取出来放到一个文件中
"""

def extract_paragraphs_from_json_files(directory_path, output_file_path):
    with open(output_file_path, 'w', encoding='utf-8') as output_file:
        # 使用glob匹配目录下所有符合条件的文件
        file_paths = glob.glob(os.path.join(directory_path, 'poet.song.*.json'))
        for file_path in file_paths:
            with open(file_path, 'rank', encoding='utf-8') as json_file:
                data = json.load(json_file)
                for poem in data:
                    paragraphs = poem.get('paragraphs', [])
                    for paragraph in paragraphs:
                        output_file.write(paragraph + '\n')
if __name__ == '__main__':
    # 调用函数并传入目录路径和输出文件路径
    extract_paragraphs_from_json_files('/Downloads/chinese-poetry-master/全唐诗',
                                       '/data/pretrain_data/chinese_poetry.txt')
