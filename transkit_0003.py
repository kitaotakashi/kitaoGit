import csv
import pprint
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import copy

from scipy import signal
from scipy.interpolate import interp1d
from scipy import stats

def lowpass(x, samplerate, fp, fs, gpass, gstop):
    fn = samplerate / 2                           #ナイキスト周波数
    wp = fp / fn                                  #ナイキスト周波数で通過域端周波数を正規化
    ws = fs / fn                                  #ナイキスト周波数で阻止域端周波数を正規化
    N, Wn = signal.buttord(wp, ws, gpass, gstop)  #オーダーとバターワースの正規化周波数を計算
    b, a = signal.butter(N, Wn, "low")           #フィルタ伝達関数の分子と分母を計算
    y = signal.filtfilt(b, a, x)                  #信号に対してフィルタをかける
    return y
                                          #フィルタ後の信号を返す
def main():
    sub_num = 1
    sub_f = 0
    sub_e = 0
    mode_num = 5#G M N A B
    trial_num = 5
    file_num = int(mode_num*trial_num)

    sample_f = 60
    dt = (float)(1/sample_f)    #[s]
    before = 6#frame before*dt->s
    after = 120#frame before*dt->s
    N = before + after
    t = np.arange(-1*before*dt, N*dt-1*before*dt, dt) # 時間 [s]

    root = "./data/transkit_0003/"

    filepath = root + "mode.csv"
    #mode_data = np.loadtxt(filepath,delimiter = ",")
    with open(filepath) as f:
        reader = csv.reader(f)
        l = [row for row in reader]
    l_T = [list(x) for x in zip(*l)]
    #print(l_T)
    #print(l_T[0])
    mode_data = l_T
    mode_label =['G','M','N','A','B']

    filepath = root + "timing.csv"
    timing_data = np.loadtxt(filepath,delimiter = ",")
    #print(timing_data[23,0])

    data_raw = []
    #for i in range(sub_num):
    for i in range(sub_f,sub_e+1,1):
        a = []
        for ii in range(file_num):
            filepath = root  + "csv/sub" + str(i+1) + "_" + str(ii+1) + ".csv"
            df = pd.read_csv(filepath, header=1,usecols=[2])
            #print(df)
            a.append(df.values)
        data_raw.append(a)

    ##>data_raw
    #print(a[24][3])#filenum,row
    #print(data_row[0][3])#sub,filenum,row#

    #データを切り取り
    #flashのタイミングに合わせてその(前)後_frame取得
    data_round = []
    #for i in range(sub_num):
    for i in range(sub_f,sub_e+1,1):
        b = []
        for ii in range(file_num):
            input = np.copy(data_raw[i][ii])#浅いコピー：data_rawとinputは独立
            #f_timing = 500
            f_timing = int(timing_data[ii,i])
            b.append(input[f_timing-before:f_timing+after])
        data_round.append(b)

    #データ整形
    #for i in range(sub_num):
    for i in range(sub_f,sub_e+1,1):
        for ii in range(file_num):
            input = data_round[i][ii]#参照渡し：連動
            for iii in range(N):
                if input[iii][0]=='---':
                    input[iii]=0
                elif type(input[iii][0])==str:
                    input[iii][0]=float(input[iii][0])
                else:
                    pass
    ## data_round
    #print(data_round[0][24])#sub,filenum,row
    #print(data_round[0][9])#sub,filenum,row

    #処理
    ##########
    #ノイズ処理
    ##########
    #処理しない
    data_round_filtered = np.copy(data_round)

    """
    fp = 3
    fs = 10
    gpass = 3       #通過域端最大損失[dB]
    gstop = 40      #阻止域端最小損失[dB]

    if mode == 0:
        pre_data_low = []
        post_data_low = []
        for i in range(len(pre_data_round[:,0])):
            y1 = lowpass(pre_data_round[i,:],1000,fp,fs,gpass,gstop)
            y2 = lowpass(post_data_round[i,:],1000,fp,fs,gpass,gstop)
            if i == 0:
                pre_data_low = y1
                post_data_low = y2
            else:
                pre_data_low = np.vstack([pre_data_low,y1])
                post_data_low = np.vstack([post_data_low,y2])
        """
    ##>data_round_filtered

    ################
    #変化率に変換：x/刺激前直径
    ################
    #刺激前直径の算出
    #変換
    data_round_rate = np.copy(data_round_filtered)
    #print(data_round)
    for i in range(sub_f,sub_e+1,1):
        for ii in range(file_num):
            base_r = data_round[i][ii][before]#flash時の直径
            if base_r == 0:
                base_r=10
            for iii in range(len(data_round_rate[i][ii])):
                data_round_rate[i][ii][iii]=data_round_rate[i][ii][iii]/base_r
    ##>data_round_rate

    ################
    #modeごとに波形平均
    ################
    #data_mode[0].append(data_round[0][1])
    #data_mode[0].append(0)
    #data_mode[0]=np.array(data_mode[0])
    #print(data_mode[0][0])
    #print(mode_data)

    #input選択
    input_tmp = np.copy(data_round_rate)#変化率を用いる
    #input_tmp = np.copy(data_round)#処理前データ

    data_mode_mean = []
    #for i in range(sub_num):
    for i in range(sub_f,sub_e+1,1):
        data_mode_mean_tmp =[]
        data_mode_tmp=[[],[],[],[],[]]
        mode_count = [0,0,0,0,0]
        for ii in range(file_num):
            idx  = mode_label.index(mode_data[i][ii])
            #print(idx)
            #データ除外
            if data_round[i][ii][before]==0:
                pass
            else:
                data_mode_tmp[idx].append(input_tmp[i][ii])
                mode_count[idx]+=1

        for ii in range(mode_num):
            tmp = data_mode_tmp[ii][0]
            for iii in range(1,mode_count[ii],1):
                tmp = np.hstack([tmp,data_mode_tmp[ii][iii]])
            tmp = tmp.mean(axis=1)#subごとのmodeごとの平均
            data_mode_mean_tmp.append(tmp)

        #print(data_mode_mean_tmp[0])
        data_mode_mean.append(data_mode_mean_tmp)

    ##############
    #グラフ化1
    ##############
    #plt.figure()
    #plt.plot(t[:],data_round[0][1])#raw:sub,filenum
    #plt.plot(t[:],data_mode_mean[0][4])#mean:sub

    plt.figure()
    subject = 0
    plt.subplot(2,3,1)
    plt.plot(t[:],data_mode_mean[subject][0])#mean:sub,mode
    plt.subplot(2,3,2)
    plt.plot(t[:],data_mode_mean[subject][1])#mean:sub,mode
    plt.subplot(2,3,3)
    plt.plot(t[:],data_mode_mean[subject][2])#mean:sub,mode
    plt.subplot(2,3,4)
    plt.plot(t[:],data_mode_mean[subject][3])#mean:sub,mode
    plt.subplot(2,3,5)
    plt.plot(t[:],data_mode_mean[subject][4])#mean:sub,mode
    plt.show()
    """
    plt.figure()
    if mode == 0:
        plt.plot(t[wnd_s:wnd_e],pre_data_sum_ave,label="pre")
        plt.plot(t[wnd_s:wnd_e],post_data_sum_ave,label="post")

        plt.title("")
        plt.xlabel("t[ms]")
        #plt.ylabel("")
        plt.legend(bbox_to_anchor=(1, 1), loc='upper right', borderaxespad=0, fontsize=10)
        plt.show()

    elif mode==1:
        plt.subplot(2,2,1)
        plt.plot(t[wnd_s:wnd_e],pre_data1_sum_ave,label="pre")
        plt.plot(t[wnd_s:wnd_e],post_data1_sum_ave,label="post")

        plt.subplot(2,2,2)
        plt.plot(t[wnd_s:wnd_e],pre_data2_sum_ave,label="pre")
        plt.plot(t[wnd_s:wnd_e],post_data2_sum_ave,label="post")

        plt.subplot(2,2,3)
        plt.plot(t[wnd_s:wnd_e],pre_data3_sum_ave,label="pre")
        plt.plot(t[wnd_s:wnd_e],post_data3_sum_ave,label="post")

        plt.subplot(2,2,4)
        plt.plot(t[wnd_s:wnd_e],pre_data4_sum_ave,label="pre")
        plt.plot(t[wnd_s:wnd_e],post_data4_sum_ave,label="post")

        #plt.plot(t[wnd_s:wnd_e],pre_data1_low[0,:],label="test")
        #plt.plot(t[wnd_s:wnd_e],post_data1_low[0,:],label="test")

        plt.title("")
        plt.xlabel("t[ms]")
        #plt.ylabel("")
        plt.legend(bbox_to_anchor=(1, 1), loc='upper right', borderaxespad=0, fontsize=10)
        plt.show()
    """

if __name__ == '__main__':
    main()
