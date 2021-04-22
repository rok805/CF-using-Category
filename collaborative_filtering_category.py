from sklearn.model_selection import StratifiedKFold
from scipy.spatial.distance import squareform, pdist
from numpy.linalg import norm
from math import isnan, isinf
from tqdm import tqdm
import pandas as pd
import numpy as np
import pickle
import time
import math
import os

#%% distance method.
def sim_cos(u,v):
    
    ind=np.where((1*(u!=0)+1*(v!=0))==2)[0]
    if len(ind) > 0:
        up = sum(u[ind] * v[ind])
        down = norm(u[ind]) * norm(v[ind])
        cos_sim = up/down
        if not math.isnan(cos_sim):
            return cos_sim
        else:
            return 0
    else:
        return 0

def sim_pcc(u,v):
    ind=np.where((1*(u!=0)+1*(v!=0))==2)[0]
    
    if len(ind)>1:
        u_m = np.mean(u[ind])
        v_m = np.mean(v[ind])
        pcc = np.sum((u[ind]-u_m)*(v[ind]-v_m)) / (norm(u[ind]-u_m)*norm(v[ind]-v_m)) # range(-1,1)
        if not isnan(pcc): # case: negative denominator
            return pcc
        else:
            return 0
    else:
        return 0

def sim_msd(u,v):
    ind=np.where((1*(u!=0)+1*(v!=0))==2)[0]
    
    if len(ind)>0:
        msd_sim = 1 - np.sum((u[ind]/5-v[ind]/5)**2)/len(ind)
        if not isnan(msd_sim):
            return msd_sim
        else:
            return 0
    else:
        return 0

def sim_jac(u,v):
    ind1=np.where((1*(u==0)+1*(v==0))==0)[0]
    ind2=np.where((1*(u==0)+1*(v==0))!=2)[0]
    return len(ind1)/len(ind2)

def sim_jmsd(u,v):
    ind1=np.where((1*(u==0)+1*(v==0))==0)[0] # 교집합
    ind2=np.where((1*(u==0)+1*(v==0))!=2)[0] # 합집합
    if len(ind1)>0:
        return (len(ind1)/len(ind2)) * (1-np.sum((u[ind1]/5-v[ind1]/5)**2)/len(ind1))
    else:
        return 0

def sim_gpsim(u,v):
    ind=np.where((1*(u==0)+1*(v==0))!=2)[0] # 공통평가장르인덱스 합집합
    u_total=sum(u[ind]) # u의 평가장르총개수
    v_total=sum(v[ind]) # v의 평가장르총개수
    u_f=u[ind]/u_total
    v_f=v[ind]/v_total
    gpsim = 1 - sum(np.maximum(u_f,v_f)*((u_f - v_f)**2)) / sum(np.maximum(u_f,v_f))
    return gpsim

def sim_gpsim_asym(u,v):
    ind=np.where((1*(u==0)+1*(v==0))!=2)[0]  # 공통평가장르인덱스 합집합
    ind2=np.where((1*(u==0)+1*(v==0))==0)[0] # 공통평가장르인덱스 교집합
    ind3=np.where((1*(v==0))==0)[0]          # v 평가장르인덱스
    u_total=sum(u[ind]) # u의 평가장르총개수
    v_total=sum(v[ind]) # v의 평가장르총개수
    u_f=u[ind]/u_total
    v_f=v[ind]/v_total
    gpsim = (1 - sum(u_f*((u_f - v_f)**2)) / sum(u_f))
    return gpsim

def sim_gpsim_mean(genre_mat, genre_mat_mean, n_user):
    sim=np.zeros([n_user, n_user])
    n=0
    z=5
    for u in tqdm(range(n_user)):
        n+=1
        for v in range(n, n_user):
            u_g=genre_mat[u,:] # 장르개수
            v_g=genre_mat[v,:]
            u_gm=genre_mat_mean[u,:] # 장르평점평균
            v_gm=genre_mat_mean[v,:]
            
            ind=np.where((1*(u_g==0)+1*(v_g==0))!=2)[0] # 공통평가장르인덱스 합집합
            u_total=sum(u_g[ind]) # u의 평가장르총개수
            v_total=sum(v_g[ind]) # v의 평가장르총개수
            sim[u,v] = 1 - sum(np.maximum(u_g[ind]/u_total,v_g[ind]/v_total) * ((u_g[ind]/u_total - v_g[ind]/v_total)**2) * (((u_gm[ind] - v_gm[ind])/z)**2) ) / sum(np.maximum(u_g[ind]/u_total,v_g[ind]/v_total))
    n=0
    for i in range(n_user):
        n+=1
        for j in range(n, n_user):
            sim[j,i] = sim[i,j]
    return sim

def sim_gpsim_mean_asym(genre_mat, genre_mat_mean, n_user):
    sim=np.zeros([n_user, n_user])
    z=5
    for u in tqdm(range(n_user)):
        for v in range(n_user):
            if u == v:
                continue
            u_g=genre_mat[u,:] # 선택장르
            v_g=genre_mat[v,:]
            u_gm=genre_mat_mean[u,:] # 장르평점평균
            v_gm=genre_mat_mean[v,:]
            
            ind=np.where((1*(u_g==0)+1*(v_g==0))!=2)[0] # 공통평가장르인덱스 합집합
            ind2=np.where((1*(u_g==0)+1*(v_g==0))==0)[0] # 공통평가장르인덱스 교집합
            ind3=np.where(1*(v_g==0)==0)[0] # v 평가장르인덱스
            u_total=sum(u_g[ind]) # u의 평가장르총개수
            v_total=sum(v_g[ind]) # v의 평가장르총개수
            sim[u,v] = 1 - sum(u_g[ind]/u_total * ((u_g[ind]/u_total - v_g[ind]/v_total)**2) * (((u_gm[ind] - v_gm[ind])/z)**2) ) / sum(u_g[ind]/u_total)
            
    return sim

def sim_gpsim_mean_abs(genre_mat, genre_mat_mean, n_user):
    sim=np.zeros([n_user, n_user])
    z=5
    n=0
    for u in tqdm(range(n_user)):
        n+=1
        for v in range(n, n_user):
            u_g=genre_mat[u,:] # 장르개수
            v_g=genre_mat[v,:]
            u_gm=genre_mat_mean[u,:] # 장르평점평균
            v_gm=genre_mat_mean[v,:]
            
            ind=np.where((1*(u_g==0)+1*(v_g==0))!=2)[0] # 공통평가장르인덱스 합집합
            u_total=sum(u_g[ind]) # u의 평가장르총개수
            v_total=sum(v_g[ind]) # v의 평가장르총개수
            sim[u,v] = 1 - sum(np.maximum(u_g[ind]/u_total,v_g[ind]/v_total) * ((u_g[ind]/u_total - v_g[ind]/v_total)**2) * (abs(u_gm[ind] - v_gm[ind])/z) ) / sum(np.maximum(u_g[ind]/u_total,v_g[ind]/v_total))
    n=0
    for i in range(n_user):
        n+=1
        for j in range(n, n_user):
            sim[j,i] = sim[i,j]
    return sim

def sim_gpsim_mean_abs_asym(genre_mat, genre_mat_mean, n_user):
    sim=np.zeros([n_user, n_user])
    z=5
    for u in tqdm(range(n_user)):
        for v in range(n_user):
            if u == v:
                continue
            u_g=genre_mat[u,:] # 장르개수
            v_g=genre_mat[v,:]
            u_gm=genre_mat_mean[u,:] # 장르평점평균
            v_gm=genre_mat_mean[v,:]
            
            ind=np.where((1*(u_g==0)+1*(v_g==0))!=2)[0] # 공통평가장르인덱스 합집합
            ind2=np.where((1*(u_g==0)+1*(v_g==0))==0)[0] # 공통평가장르인덱스 교집합
            ind3=np.where(1*(v_g==0)==0)[0] # v 평가장르인덱스
            u_total=sum(u_g[ind]) # u의 평가장르총개수
            v_total=sum(v_g[ind]) # v의 평가장르총개수
            sim[u,v] = 1 - sum(u_g[ind]/u_total * ((u_g[ind]/u_total - v_g[ind]/v_total)**2) * (abs(u_gm[ind] - v_gm[ind])/z) ) / sum(u_g[ind]/u_total)
    
    return sim

def sim_gpsim_mean2(genre_mat, genre_mat_mean, data_d_trn_data_mean, n_user):
    sim=np.zeros([n_user, n_user])
    z=5
    n=0
    for u in tqdm(range(n_user)):
        n+=1
        for v in range(n, n_user):
            u_g=genre_mat[u,:] # 장르 평가 개수 벡터
            v_g=genre_mat[v,:]
            u_gm=genre_mat_mean[u,:] # 장르평점평균
            v_gm=genre_mat_mean[v,:]
           
            ind=np.where((1*(u_g==0)+1*(v_g==0))!=2)[0] # 공통평가장르인덱스 합집합
            u_total=sum(u_g[ind]) # u의 평가장르총개수
            v_total=sum(v_g[ind]) # v의 평가장르총개수
            sim[u,v] = 1 - sum(np.maximum(u_g[ind]/u_total,v_g[ind]/v_total) * ((u_g[ind]/u_total - v_g[ind]/v_total)**2) * (((u_gm[ind] - v_gm[ind])/z)**2) * (((u_gm[ind]-data_d_trn_data_mean[u])-(v_gm[ind]-data_d_trn_data_mean[v]))/(2*z))**2)  / sum(np.maximum(u_g[ind]/u_total,v_g[ind]/v_total))
    n=0
    for i in range(n_user):
        n+=1
        for j in range(n, n_user):
            sim[j,i] = sim[i,j]
    return sim

def sim_gpsim_mean2_asym(genre_mat, genre_mat_mean, data_d_trn_data_mean, n_user):
    sim=np.zeros([n_user, n_user])
    z=5
    for u in tqdm(range(n_user)):
        for v in range(n_user):
            if u == v:
                continue
            u_g=genre_mat[u,:] # 장르 평가 개수 벡터
            v_g=genre_mat[v,:]
            u_gm=genre_mat_mean[u,:] # 장르평점평균
            v_gm=genre_mat_mean[v,:]
           
            ind=np.where((1*(u_g==0)+1*(v_g==0))!=2)[0] # 공통평가장르인덱스 합집합
            ind2=np.where((1*(u_g==0)+1*(v_g==0))==0)[0] # 공통평가장르인덱스 교집합
            ind3=np.where(1*(v_g==0)==0)[0] # v 평가장르인덱스
            u_total=sum(u_g[ind]) # u의 평가장르총개수
            v_total=sum(v_g[ind]) # v의 평가장르총개수
            sim[u,v] = (1 - sum(u_g[ind]/u_total * ((u_g[ind]/u_total - v_g[ind]/v_total)**2) * (((u_gm[ind] - v_gm[ind])/z)**2) * (((u_gm[ind]-data_d_trn_data_mean[u])-(v_gm[ind]-data_d_trn_data_mean[v]))/(2*z))**2)  / sum(u_g[ind]/u_total)) * ((len(ind2)+1)/(len(ind3)+1))
    
    return sim

def sim_gpsim_mean2_abs(genre_mat, genre_mat_mean, data_d_trn_data_mean, n_user):
    sim=np.zeros([n_user, n_user])
    z=5
    n=0
    for u in tqdm(range(n_user)):
        n+=1
        for v in range(n, n_user):
            u_g=genre_mat[u,:] # 장르 평가 개수 벡터
            v_g=genre_mat[v,:]
            u_gm=genre_mat_mean[u,:] # 장르평점평균
            v_gm=genre_mat_mean[v,:]
           
            ind=np.where((1*(u_g==0)+1*(v_g==0))!=2)[0] # 공통평가장르인덱스 합집합
            u_total=sum(u_g[ind]) # u의 평가장르총개수
            v_total=sum(v_g[ind]) # v의 평가장르총개수
            sim[u,v] = 1 - sum(np.maximum(u_g[ind]/u_total,v_g[ind]/v_total) * ((u_g[ind]/u_total - v_g[ind]/v_total)**2) * (abs(u_gm[ind] - v_gm[ind])/z) * (abs((u_gm[ind]-data_d_trn_data_mean[u])-(v_gm[ind]-data_d_trn_data_mean[v]))/(2*z)))  / sum(np.maximum(u_g[ind]/u_total,v_g[ind]/v_total))
    n=0
    for i in range(n_user):
        n+=1
        for j in range(n, n_user):
            sim[j,i] = sim[i,j]
    return sim

def sim_gpsim_mean2_abs_asym(genre_mat, genre_mat_mean, data_d_trn_data_mean, n_user):
    sim=np.zeros([n_user, n_user])
    z=5
    for u in tqdm(range(n_user)):
        for v in range(n_user):
            if u == v:
                continue
            u_g=genre_mat[u,:] # 장르 평가 개수 벡터
            v_g=genre_mat[v,:]
            u_gm=genre_mat_mean[u,:] # 장르평점평균
            v_gm=genre_mat_mean[v,:]
           
            ind=np.where((1*(u_g==0)+1*(v_g==0))!=2)[0] # 공통평가장르인덱스 합집합
            ind2=np.where((1*(u_g==0)+1*(v_g==0))==0)[0] # 공통평가장르인덱스 교집합
            ind3=np.where(1*(v_g==0)==0)[0] # v 평가장르인덱스
            u_total=sum(u_g[ind]) # u의 평가장르총개수
            v_total=sum(v_g[ind]) # v의 평가장르총개수
            sim[u,v] = (1 - sum(u_g[ind]/u_total * ((u_g[ind]/u_total - v_g[ind]/v_total)**2) * (abs(u_gm[ind] - v_gm[ind])/z) * (abs((u_gm[ind]-data_d_trn_data_mean[u])-(v_gm[ind]-data_d_trn_data_mean[v]))/(2*z)))  / sum(u_g[ind]/u_total)) * ((len(ind2)+1)/(len(ind3)+1))

    return sim

def sim_gpsim_mean3(genre_mat, genre_mat_mean, data_d_trn_data_mean, n_user):
    sim=np.zeros([n_user, n_user])
    z=5
    n=0
    for u in tqdm(range(n_user)):
        n+=1
        for v in range(n, n_user):
            u_g=genre_mat[u,:] # 장르 평가 개수 벡터
            v_g=genre_mat[v,:]
            u_gm=genre_mat_mean[u,:] # 장르평점평균
            v_gm=genre_mat_mean[v,:]
           
            ind=np.where((1*(u_g==0)+1*(v_g==0))!=2)[0] # 공통평가장르인덱스 합집합
            u_total=sum(u_g[ind]) # u의 평가장르총개수
            v_total=sum(v_g[ind]) # v의 평가장르총개수
            sim[u,v] = 1 - sum(np.maximum(u_g[ind]/u_total,v_g[ind]/v_total) * ((u_g[ind]/u_total - v_g[ind]/v_total)**2) * (((u_gm[ind]-data_d_trn_data_mean[u])-(v_gm[ind]-data_d_trn_data_mean[v]))/(2*z))**2)  / sum(np.maximum(u_g[ind]/u_total,v_g[ind]/v_total))
    n=0
    for i in range(n_user):
        n+=1
        for j in range(n, n_user):
            sim[j,i] = sim[i,j]
    return sim

def sim_gpsim_mean3_asym(genre_mat, genre_mat_mean, data_d_trn_data_mean, n_user, c):
    sim=np.zeros([n_user, n_user])
    z=5
    for u in tqdm(range(n_user)):
        for v in range(n_user):
            if u == v:
                continue
            u_g=genre_mat[u,:] # 장르 평가 개수 벡터
            v_g=genre_mat[v,:]
            u_gm=genre_mat_mean[u,:] # 장르평점평균
            v_gm=genre_mat_mean[v,:]
           
            ind=np.where((1*(u_g==0)+1*(v_g==0))!=2)[0] # 공통평가장르인덱스 합집합
            ind2=np.where((1*(u_g==0)+1*(v_g==0))==0)[0] # 공통평가장르인덱스 교집합
            ind3=np.where(1*(v_g==0)==0)[0] # v 평가장르인덱스
            u_total=sum(u_g[ind]) # u의 평가장르총개수
            v_total=sum(v_g[ind]) # v의 평가장르총개수
            sim[u,v] = 1 - sum(u_g[ind]/u_total * ((u_g[ind]/u_total - v_g[ind]/v_total)**2) * (((((u_gm[ind]-data_d_trn_data_mean[u])-(v_gm[ind]-data_d_trn_data_mean[v]))/(2*z))**2)*c) )  / sum(u_g[ind]/u_total)

    return sim

def sim_gpsim_mean3_abs(genre_mat, genre_mat_mean, data_d_trn_data_mean, n_user):
    sim=np.zeros([n_user, n_user])
    z=5
    n=0
    for u in tqdm(range(n_user)):
        n+=1
        for v in range(n, n_user):
            u_g=genre_mat[u,:] # 장르 평가 개수 벡터
            v_g=genre_mat[v,:]
            u_gm=genre_mat_mean[u,:] # 장르평점평균
            v_gm=genre_mat_mean[v,:]
           
            ind=np.where((1*(u_g==0)+1*(v_g==0))!=2)[0] # 공통평가장르인덱스 합집합
            u_total=sum(u_g[ind]) # u의 평가장르총개수
            v_total=sum(v_g[ind]) # v의 평가장르총개수
            sim[u,v] = 1 - sum(np.maximum(u_g[ind]/u_total,v_g[ind]/v_total) * ((u_g[ind]/u_total - v_g[ind]/v_total)**2) * (abs((u_gm[ind]-data_d_trn_data_mean[u])-(v_gm[ind]-data_d_trn_data_mean[v]))/(2*z)))  / sum(np.maximum(u_g[ind]/u_total,v_g[ind]/v_total))
    n=0
    for i in range(n_user):
        n+=1
        for j in range(n, n_user):
            sim[j,i] = sim[i,j]
    return sim

def sim_gpsim_mean3_abs_asym(genre_mat, genre_mat_mean, data_d_trn_data_mean, n_user):
    sim=np.zeros([n_user, n_user])
    z=5
    for u in tqdm(range(n_user)):
        for v in range(n_user):
            u_g=genre_mat[u,:] # 장르 평가 개수 벡터
            v_g=genre_mat[v,:]
            u_gm=genre_mat_mean[u,:] # 장르평점평균
            v_gm=genre_mat_mean[v,:]
           
            ind=np.where((1*(u_g==0)+1*(v_g==0))!=2)[0] # 공통평가장르인덱스 합집합
            ind2=np.where((1*(u_g==0)+1*(v_g==0))==0)[0] # 공통평가장르인덱스 교집합
            ind3=np.where(1*(v_g==0)==0)[0] # u 평가장르인덱스
            u_total=sum(u_g[ind]) # u의 평가장르총개수
            v_total=sum(v_g[ind]) # v의 평가장르총개수
            sim[u,v] = 1 - sum(u_g[ind]/u_total * ((u_g[ind]/u_total - v_g[ind]/v_total)**2) * (abs((u_gm[ind]-data_d_trn_data_mean[u])-(v_gm[ind]-data_d_trn_data_mean[v]))/(2*z)))  / sum(u_g[ind]/u_total)

    return sim


#%% 데이터 불러오기 및 rating, item 데이터 전처리.
data_name = 'MovieLens100K'
cwd=os.getcwd()

if data_name == 'MovieLens100K': # MovieLens100K load and preprocessing
    
    # MovieLens100K: u.data, item.txt 의 경로
    data = pd.read_table(os.path.join(cwd,'movielens\\order\\u.data'),header=None, names=['uid','iid','r','ts'])
    item = pd.read_table(os.path.join(cwd,'movielens\\order\\item.txt'), sep='|', header=None)
    item_cols=['movie id','movie title','release date','video release date','IMDb URL','unknown','Action','Adventure','Animation',"Children's",'Comedy','Crime','Documentary','Drama','Fantasy','Film-Noir','Horror','Musical','Mystery','Romance','Sci-Fi','Thriller','War','Western']
    item.columns = item_cols
    item=np.array(item.drop(columns=['movie id','movie title', 'release date', 'video release date', 'IMDb URL', 'unknown']))
    # uid, iid minus one. 평점 행렬의 행, 열 지정을 위해서 -1 을 함.
    data['uid'] = np.array(data.uid) - 1
    data['iid'] = np.array(data.iid) - 1

elif data_name == 'MovieLens1M': # MovieLens1M load and preprocessing
    # MovieLens1M: rating.dat, movies.dat 의 경로
    data = pd.read_csv(os.path.join(cwd,'movielens\\1M\\ratings.dat'), sep='::', header=None, names=['uid','iid','r','ts']).drop(columns=['ts'])
    item = pd.read_table(os.path.join(cwd, 'movielens\\1M\\movies.dat'), sep='::', header=None)
    
    # item indexing
    m_d = {}
    for n, i in enumerate(item[0]):
        m_d[i] = n
    item[0] = sorted(m_d.values())
    
    i_to_n = []
    for i in range(data.shape[0]):
        i_to_n.append(m_d[data.loc[i,'iid']])
    data['iid'] = i_to_n
    
    # uid minus one. 평점 행렬의 행, 열 지정을 위해서 -1 을 함.
    data['uid'] = np.array(data.uid) - 1
    
    # movie genre matrix
    genre_name = set()
    for i in range(item.shape[0]):
        gg = item.loc[i,2].split('|')
        for j in gg:
            genre_name.add(j)
    
    m_to_idx = {}
    for n, i in enumerate(genre_name):
        m_to_idx[i] = n
    i_to_g = {}
    for i in range(item.shape[0]):
        i_to_g[i] = [m_to_idx[g] for g in item.loc[i,2].split("|")]
    i_genre = np.zeros([item.shape[0], len(genre_name)])
    for i in i_to_g:
        for j in i_to_g[i]:
            i_genre[i,j] = 1
    item = i_genre

elif data_name == 'Netflix': # Netflix load and preprocessing
    # Netflix: ratings.csv, movies.csv 의 경로
    data = pd.read_csv(os.path.join(cwd,'netflix\\movie_ratings\\ratings.csv'), header=0, names=['uid','iid','r','ts']).drop(columns=['ts'])
    item = pd.read_table(os.path.join(cwd,'netflix\\movie_ratings\\movies.csv'), sep=',', header=0)
    item.columns=[0,1,2]
    # item indexing
    m_d = {}
    for n, i in enumerate(item[0]):
        m_d[i] = n
    item[0] = sorted(m_d.values())
    
    i_to_n = []
    for i in range(data.shape[0]):
        i_to_n.append(m_d[data.loc[i,'iid']])
    data['iid'] = i_to_n
    
    # uid minus one. 평점 행렬의 행, 열 지정을 위해서 -1 을 함.
    data['uid'] = np.array(data.uid) - 1
    
    # movie genre matrix
    genre_name = set()
    for i in range(item.shape[0]):
        gg = item.loc[i,2].split('|')
        for j in gg:
            genre_name.add(j)
    
    m_to_idx = {}
    for n, i in enumerate(genre_name):
        m_to_idx[i] = n
    i_to_g = {}
    for i in range(item.shape[0]):
        i_to_g[i] = [m_to_idx[g] for g in item.loc[i,2].split("|")]
    i_genre = np.zeros([item.shape[0], len(genre_name)])
    for i in i_to_g:
        for j in i_to_g[i]:
            i_genre[i,j] = 1
    item = i_genre




#%%
# Collaborative Filtering
##########################################################################################################################################################
##########################################################################################################################################################
##########################################################################################################################################################

#%% 데이터 분할.
# cv validation, random state, split setting.
cv = 5
rs = 35
sk = StratifiedKFold(n_splits=cv, random_state=rs, shuffle=True)

# 결과저장 데이터프레임
result_mae_rmse = pd.DataFrame(columns=['fold','k','MAE','RMSE'])
result_topN = pd.DataFrame(columns=['fold','k','Precision','Recall','F1_score'])
count = 0


# 실험.
cross_val=True # cross validation 사용. 
result_d = {}
alpha=0.9
sim_name = 'gpsim_mean3_asym'
cc=1 # constant for GPSIM_asym_mean3.

# split dataset
for f, (trn,val) in enumerate(sk.split(data,data['uid'].values)):
    print()
    print(f'cv: {f+1}')
    trn_data = data.iloc[trn]
    val_data = data.iloc[val]

    # train dataset rating dictionary.
    data_d_trn_data = {}
    for u, i, r in zip(trn_data['uid'], trn_data['iid'], trn_data['r']):
        if u not in data_d_trn_data:
            data_d_trn_data[u] = {i:r}
        else:
            data_d_trn_data[u][i] = r

    # train dataset user rating mean dictionary.
    data_d_trn_data_mean = {}
    for u in data_d_trn_data:
        data_d_trn_data_mean[u] = np.mean(list(data_d_trn_data[u].values()))
    

    #%% rating matrix about train/test set.

    n_item = item.shape[0]
    n_user = len(set(data['uid']))

    # train rating matrix
    rating_matrix = np.zeros((n_user, n_item))
    for u, i, r in zip(trn_data['uid'], trn_data['iid'], trn_data['r']):
        rating_matrix[u,i] = r

    # test rating matrix
    rating_matrix_test = np.zeros((n_user, n_item))
    for u, i, r in zip(val_data['uid'], val_data['iid'], val_data['r']):
        rating_matrix_test[u,i] = r


    #%% genre matrix.
    # user genre dictionary. 사용자의 장르의 개수와 장르에 대한 평균.
    genre_d = {}
    for u in set(data['uid']):
        genre_d[u]={}
        for g in range(item.shape[1]):
            genre_d[u][g]=[0,0,0]
    
    for i in range(data.shape[0]):
        ui = data.loc[i,'uid']
        ii = data.loc[i,'iid']
        iir = data.loc[i,'r']
        
        ii_genre = np.where(item[ii,:] != 0)[0] # 장르 인덱스.
        for g in ii_genre:
            genre_d[ui][g][0]+=1
            genre_d[ui][g][1]+=iir
    
    genre_mat = np.zeros([n_user, item.shape[1]]) # 장르개수 matrix
    for u in range(n_user):
        for g in genre_d[u]:
            genre_mat[u,g] = genre_d[u][g][0]
    
    genre_mat_mean = np.zeros([n_user, item.shape[1]]) # 장르평균 matrix
    for u in range(n_user):
        for g in genre_d[u]:
            try:
                genre_mat_mean[u,g] = genre_d[u][g][1]/genre_d[u][g][0]
            except ZeroDivisionError:
                genre_mat_mean[u,g] = 0
                
    #%% 1. similarity calculation.

    print('\n')
    print(f'similarity calculation: {sim_name}')

    s=time.time()
    
    # 기본적인 유사도지표
    if sim_name=='cos':    
        sim=pdist(rating_matrix,metric=sim_cos)
        sim=squareform(sim)
    elif sim_name=='pcc':
        sim=pdist(rating_matrix,metric=sim_pcc)
        sim=squareform(sim)
    elif sim_name=='msd':
        sim=pdist(rating_matrix,metric=sim_msd)
        sim=squareform(sim)
    elif sim_name=='jmsd':
        sim=pdist(rating_matrix,metric=sim_jmsd)
        sim=squareform(sim)
    
    # 참조논문 유사도지표
    elif sim_name=='gpsim':    
        sim=pdist(genre_mat,metric=sim_gpsim)
        sim=squareform(sim)
    elif sim_name=='gpsim_pcc':
        sim1=pdist(genre_mat,metric=sim_gpsim)
        sim2=pdist(rating_matrix,metric=sim_pcc)
        sim1=squareform(sim1)
        sim2=squareform(sim2)
        sim=alpha*sim1 + (1-alpha)*sim2
    elif sim_name=='gpsim_jmsd':
        sim1=pdist(genre_mat,metric=sim_gpsim)
        sim2=pdist(rating_matrix,metric=sim_jmsd)
        sim1=squareform(sim1)
        sim2=squareform(sim2)
        sim=alpha*sim1 + (1-alpha)*sim2
    
    # 제안 유사도지표
    elif sim_name=='gpsim_asym':  # GPSIM_ASYM
        sim = sim_gpsim_asym(genre_mat, genre_mat_mean, n_user)
        
    elif sim_name=='gpsim_mean':  # GPSIM_mean
        sim=sim_gpsim_mean(genre_mat, genre_mat_mean, n_user)
    elif sim_name=='gpsim_mean_abs': 
        sim=sim_gpsim_mean_abs(genre_mat, genre_mat_mean, n_user)
    elif sim_name=='gpsim_mean_asym': 
        sim=sim_gpsim_mean_asym(genre_mat, genre_mat_mean, n_user)
    elif sim_name=='gpsim_mean_abs_asym': 
        sim=sim_gpsim_mean_abs_asym(genre_mat, genre_mat_mean, n_user)    
    elif sim_name=='gpsim_mean_pcc':
        sim1=sim_gpsim_mean(genre_mat, genre_mat_mean, n_user)
        sim2=pdist(rating_matrix,metric=sim_pcc)
        sim2=squareform(sim2)
        sim=alpha*sim1 + (1-alpha)*sim2
    elif sim_name=='gpsim_mean_jmsd':
        sim1=sim_gpsim_mean(genre_mat, genre_mat_mean, n_user)
        sim2=pdist(rating_matrix,metric=sim_jmsd)
        sim2=squareform(sim2)
        sim=alpha*sim1 + (1-alpha)*sim2
    elif sim_name=='gpsim_mean_abs_pcc':
        sim1=sim_gpsim_mean_abs(genre_mat, genre_mat_mean, n_user)
        sim2=pdist(rating_matrix,metric=sim_pcc)
        sim2=squareform(sim2)
        sim=alpha*sim1 + (1-alpha)*sim2
    elif sim_name=='gpsim_mean_abs_jmsd':
        sim1=sim_gpsim_mean_abs(genre_mat, genre_mat_mean, n_user)
        sim2=pdist(rating_matrix,metric=sim_jmsd)
        sim2=squareform(sim2)
        sim=alpha*sim1 + (1-alpha)*sim2

    elif sim_name=='gpsim_mean2':
        sim=sim_gpsim_mean2(genre_mat, genre_mat_mean, data_d_trn_data_mean, n_user)    
    elif sim_name=='gpsim_mean2_abs':
        sim=sim_gpsim_mean2_abs(genre_mat, genre_mat_mean, data_d_trn_data_mean, n_user)    
    elif sim_name=='gpsim_mean2_asym': 
        sim=sim_gpsim_mean2_asym(genre_mat, genre_mat_mean, data_d_trn_data_mean, n_user)    
    elif sim_name=='gpsim_mean2_abs_asym':
        sim=sim_gpsim_mean2_abs_asym(genre_mat, genre_mat_mean, data_d_trn_data_mean, n_user)
        
    elif sim_name=='gpsim_mean3': 
        sim=sim_gpsim_mean3(genre_mat, genre_mat_mean, data_d_trn_data_mean, n_user)    
    elif sim_name=='gpsim_mean3_abs': 
        sim=sim_gpsim_mean3_abs(genre_mat, genre_mat_mean, data_d_trn_data_mean, n_user)
    elif sim_name=='gpsim_mean3_asym': 
        sim=sim_gpsim_mean3_asym(genre_mat, genre_mat_mean, data_d_trn_data_mean, n_user, c=cc)
    elif sim_name=='gpsim_mean3_abs_asym': 
        sim=sim_gpsim_mean3_abs_asym(genre_mat, genre_mat_mean, data_d_trn_data_mean, n_user)
    elif sim_name=='gpsim_mean3_asym_pcc': 
        sim1=sim_gpsim_mean3_asym(genre_mat, genre_mat_mean, data_d_trn_data_mean, n_user, c=cc)
        sim2=pdist(rating_matrix,metric=sim_pcc)
        sim2=squareform(sim2)
        sim=alpha*sim1 + (1-alpha)*sim2
    elif sim_name=='gpsim_mean3_asym_jmsd':
        sim1=sim_gpsim_mean3_asym(genre_mat, genre_mat_mean, data_d_trn_data_mean, n_user, c=cc)
        sim2=pdist(rating_matrix,metric=sim_jmsd)
        sim2=squareform(sim2)
        sim=alpha*sim1 + (1-alpha)*sim2
        
    
    print(time.time()-s)
    
    
    # sel_nn, sel_sim: neighbor 100 명까지만 id와 similarity를 저장.
    np.fill_diagonal(sim,-1)
    nb_ind=np.argsort(sim,axis=1)[:,::-1] # nearest neighbor sort.
    sel_nn=nb_ind[:,:100]
    sel_sim=np.sort(sim,axis=1)[:,::-1][:,:100]


    #%% 2. prediction
    print('\n')
    print('prediction: k=10,20, ..., 100')
    rating_matrix_prediction = rating_matrix.copy()
        
    s=time.time()

    for k in tqdm([10,20,30,40,50,60,70,80,90,100]):
        
        for user in range(rating_matrix.shape[0]):
            
            for p_item in list(np.where(rating_matrix_test[user,:]!=0)[0]):
                
                molecule = []
                denominator = []
                
                #call K neighbors
                user_neighbor = sel_nn[user,:k]
                user_neighbor_sim = sel_sim[user,:k]
                
                for neighbor, neighbor_sim in zip(user_neighbor, user_neighbor_sim):
                    
                    if p_item in data_d_trn_data[neighbor].keys():
                        molecule.append(neighbor_sim * (rating_matrix[neighbor, p_item] - data_d_trn_data_mean[neighbor]))
                        denominator.append(abs(neighbor_sim))
                try:
                    rating_matrix_prediction[user, p_item] = data_d_trn_data_mean[user] + (sum(molecule) / sum(denominator))
                except ZeroDivisionError:
                    rating_matrix_prediction[user, p_item] = math.nan
        



       #%%3. performance
        # MAE, RMSE
        
        precision, recall, f1_score = [], [], []
        rec_score = 4 # 추천 기준 점수.
        pp=[]
        rr=[]
        for u, i, r in zip(val_data['uid'], val_data['iid'], val_data['r']):
            p = rating_matrix_prediction[u,i]
            if not math.isnan(p):
                pp.append(p)
                rr.append(r)
                
        d = [abs(a-b) for a,b in zip(pp,rr)]
        mae = sum(d)/len(d)
        rmse = np.sqrt(sum(np.square(np.array(d)))/len(d))
        
        result_mae_rmse.loc[count] = [f, k, mae, rmse]
        
        
        # precision, recall, f1-score
        
        pp = np.array(pp)
        rr = np.array(rr)
        TPP = len(set(np.where(pp >= rec_score)[0]).intersection(set(np.where(rr >= rec_score)[0])))
        FPP = len(set(np.where(pp >= rec_score)[0]).intersection(set(np.where(rr < rec_score)[0])))
        FNP = len(set(np.where(pp < rec_score)[0]).intersection(set(np.where(rr >= rec_score)[0])))
        
        _precision = TPP / (TPP + FPP)
        _recall = TPP / (TPP + FNP)
        _f1_score = 2 * _precision * _recall / (_precision + _recall)

        result_topN.loc[count] = [f, k, _precision, _recall, _f1_score]
        
        
        count += 1
    print(time.time() - s)
    
    # 반복여부 (cross validation)
    if cross_val == True:
        continue
    else:
        break

#%%
result_1 = result_mae_rmse.groupby(['k']).mean().drop(columns=['fold'])
result_2 = result_topN.groupby(['k']).mean().drop(columns=['fold'])
result = pd.merge(result_1, result_2, on=result_1.index).drop(columns=['key_0'])



#%% 시험결과 저장.
import pickle
import datetime
with open('result/result_{}_{}_{}.pickle'.format(str(datetime.datetime.now())[:13]+'시'+str(datetime.datetime.now())[14:16]+'분',data_name,sim_name), 'wb') as f:
    pickle.dump(result, f)
