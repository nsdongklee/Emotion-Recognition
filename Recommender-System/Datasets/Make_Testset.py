#!/usr/bin/env python
# coding: utf-8


import pandas as pd
from glob import glob
import sys
from os import rename, listdir

# 테스트셋 경로, 프레임 준비하기
data_path = glob(pathname='./data/*')
title_path = listdir(path='./data/')
title_list = [idx[ :-5] for idx in title_path]

train_set = pd.DataFrame(columns=['id', 'like_cnt', 'plylst_title',
                                   'songs', 'tags', 'updt_date'])

#print(data_path, title_list)

# 플레이리스트 테스트셋 추가하기
def append_list(frame, path_dir, title):
    
    for idx, path in enumerate(path_dir):
        df = pd.read_excel(path, index_col=0)
        df = df.rename(columns={'id':'song_id', 'id.1':'artist_id', '곡 태그':'tags'})
        
        songs = df.song_id.to_list(); print(songs)
        tags = df.tags.to_list(); print(tags)
            
        row = {'id' : idx,                            # 플레이리스트 id 부여
               'like_cnt' : 71,                       # 최빈값 71, 평균값 95
               'plylst_title' : title[idx],           # 타이틀 뭘로 할지
               'songs' : songs,                       # 각 플레이리스트의 노래id들
               'tags' : tags,                         # 각 플레이리스트의 태그정보들
               'updt_date': '2019-12-05 15:15:18.000' # 날짜 뭘로 할지 정해야함(없어도 무방)
              }
        frame = frame.append(row, ignore_index=True)
    return frame

testset = append_list(frame=train_set, path_dir=data_path, title=title_list)
testset