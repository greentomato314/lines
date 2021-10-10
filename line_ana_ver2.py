from PIL import Image, ImageFilter
import matplotlib.pyplot as plt
import numpy as np
import collections
import random

# 画像二値化(大津の手法)https://algorithm.joho.info/programming/python/opencv-otsu-thresholding-py/
def threshold_otsu(gray, min_value=0, max_value=255):
    # ヒストグラムの算出
    hist = [0]*(256)
    for i in range(gray.shape[0]):
        for j in range(gray.shape[1]):
            hist[gray[i][j]] += 1

    s_max = (0,-10)

    for th in range(256):
        
        # クラス1とクラス2の画素数を計算
        n1 = sum(hist[:th])
        n2 = sum(hist[th:])
        
        # クラス1とクラス2の画素値の平均を計算
        if n1 == 0 : mu1 = 0
        else : mu1 = sum([i * hist[i] for i in range(0,th)]) / n1   
        if n2 == 0 : mu2 = 0
        else : mu2 = sum([i * hist[i] for i in range(th, 256)]) / n2

        # クラス間分散の分子を計算
        s = n1 * n2 * (mu1 - mu2) ** 2

        # クラス間分散の分子が最大のとき、クラス間分散の分子と閾値を記録
        if s > s_max[1]:
            s_max = (th, s)
    
    # クラス間分散が最大のときの閾値を取得
    t = s_max[0]
    
    # 算出した閾値で二値化処理
    gray2 = np.copy(gray)
    gray2[gray2 < t] = min_value
    gray2[gray2 >= t] = max_value

    return gray2


class Line_analysis:
    def __init__(self,arr,mode='w'):
        '''
        arr (np.ndarray): 二値化済みの画像配列
        mode (str) : 'W' or 'b' 分析対象が白か黒か
        '''
        self.B = 255 if mode == 'w' else 0
        self.H,self.W = arr.shape
        self.arr = np.copy(arr)
        self.group_cnt = self.group_div()
    
    def group_div(self,ignore=100): 
        '''
        連結分解するところ
        ignore(int):連結個数がignore以下のものは無視する。
        '''
        used = [[False]*self.W for i in range(self.H)]
        dxy = [(1,0),(0,1),(-1,0),(0,-1)]
        self.groups = []
        for i in range(self.H):
            for j in range(self.W):
                if used[i][j] or self.arr[i][j] != self.B:
                    continue
                st = (i,j)
                d = collections.deque([st])
                group = []
                while d:
                    x,y = d.popleft()
                    if used[x][y]:
                        continue
                    group.append((x,y))
                    used[x][y] =True
                    for dx,dy in dxy:
                        if 0<=x+dx<self.H and 0<=y+dy<self.W:
                            if used[x+dx][y+dy] or self.arr[x+dx][y+dy] != self.B:
                                continue
                            d.append((x+dx,y+dy))
                if len(group) > ignore:
                    self.groups.append(group)
        self.groups.sort(key=lambda x:-len(x))
        return len(self.groups)

    def solve(self,sampling=50,trim=500):
        '''
        グループごとに線幅を求めるところ
        sampling(int): 計算に使うサンプリング数
        trim(int): トリム数(端っこいくつを落とすか)
        実際にサンプリングするのは sampling+2*trim
        '''
        sampling_all = sampling+2*trim
        self.d_calc_all = []
        self.d_calc_ave = []
        self.d_calc_mid = []
        for i in range(self.group_cnt):
            g = self.groups[i]
            nl = len(g)//3
            nr = len(g)-nl #連結成分の真ん中らへんをとる。(3分割した真ん中)
            d0 = []
            for _ in range(sampling_all):
                a,b = 0,0
                s = random.randint(nl,nr)
                st = g[s]
                d = collections.deque([st])
                while d: # 下に移動
                    x,y = d.popleft()
                    a += 1
                    if 0<=x+1<self.H and self.arr[x+1][y] == self.B:
                        d.append((x+1,y))
                d = collections.deque([st])
                while d: # 上に移動
                    x,y = d.popleft()
                    a += 1
                    if 0<=x-1<self.H and self.arr[x-1][y] == self.B:
                        d.append((x-1,y))
                d = collections.deque([st])
                while d: # 右に移動
                    x,y = d.popleft()
                    b += 1
                    if 0<=y+1<self.W and self.arr[x][y+1] == self.B:
                        d.append((x,y+1))
                d = collections.deque([st])
                while d: # 左に移動
                    x,y = d.popleft()
                    b += 1
                    if 0<=y-11<self.W and self.arr[x][y-1] == self.B:
                        d.append((x,y-1))
                a -= 1
                b -= 1
                d0.append(a*b/(a**2+b**2)**(1/2))
            d0.sort()
            d0 = d0[trim:-trim]
            darr = np.copy(np.array(d0))
            self.d_calc_all.append(darr)
            self.d_calc_ave.append(np.mean(darr))
            self.d_calc_mid.append(np.median(darr))

    def get_d_L(self,mode='ave'):
        '''
        すべてのグループの線幅の大きさとその全長を返す。
        mode(str): 代表径をどうするか(平均:ave,中央値:midを指定)
        return(list(tuple)):
        '''
        ret = []
        if mode == 'ave':
            for i in range(self.group_cnt):
                d,L = self.d_calc_ave[i],len(self.groups[i])/self.d_calc_ave[i]
                ret.append((d,L))
        elif mode == 'mid':
            for i in range(self.group_cnt):
                d,L = self.d_calc_mid[i],len(self.groups[i])/self.d_calc_mid[i]
                ret.append((d,L))
        return ret
    
    def group_arr(self,v):
        '''
        グループvの画像配列を出力
        v(int):グループのインデックス
        return (np.ndarray):
        '''
        arrg = np.zeros((self.H,self.W), dtype=np.int64)
        for x,y in self.groups[v]:
            arrg[x][y] = 255
        return arrg


if __name__ == '__main__':

    im = Image.open('data_image.jpg')
    im = im.convert('L').crop((0, 0, 800, 550))
    arr = np.array(im)
    arr = threshold_otsu(arr)
    plt.imshow(arr,cmap = "gray")
    plt.show()
    
    LA = Line_analysis(arr)
    LA.solve()
    print(LA.group_cnt)
    print(LA.get_d_L(mode='ave'))
    '''
    for i in range(LA.group_cnt):
        plt.imshow(LA.group_arr(i),cmap = "gray")
        plt.show()
    '''

