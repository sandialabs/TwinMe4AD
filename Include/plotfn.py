import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib as mpl
from matplotlib import pyplot as plt
import matplotlib.gridspec as gridspec
from mpl_toolkits.axes_grid1.inset_locator import inset_axes
from matplotlib.transforms import ScaledTranslation
from scipy.stats import gaussian_kde
import warnings
import os
from collections import defaultdict
import scipy.stats as st 
from IPython.display import clear_output

pd.options.mode.chained_assignment=None
plt.rcParams['font.family'] = 'Times New Roman'
warnings.filterwarnings('ignore')

data_path = './Data/'

healthyRHR = 100
rhrRange = [x for x in range(90, 111)]
thresholdRange = [int(x*1000) for x in np.arange(0.005, 0.021, 0.001)]

class Plotpy:
    def __init__(self, data_path, healthy_rhr, rhrRange, thresholdRange):
        self.data_path = data_path
        self.healthyRHR = healthy_rhr
        self.rhrRange = rhrRange
        self.thresholdRange = thresholdRange
        self.data = {
            "ORG":{
                    'R':os.path.join(self.data_path, f'Confusion_matrix_Threshold_all_RHR_{rhrRange[0]}_{rhrRange[-1]}_real.csv'),
                    'S':os.path.join(self.data_path, f'Confusion_matrix_Threshold_all_RHR_{rhrRange[0]}_{rhrRange[-1]}_syn.csv'),
                    'RS':os.path.join(self.data_path, f'Confusion_matrix_Threshold_all_RHR_{rhrRange[0]}_{rhrRange[-1]}_real_syn.csv')
            },
            "NOISE":{
                'R':os.path.join(self.data_path, f'Confusion_matrix_Threshold_4_RHR_{rhrRange[0]}_{rhrRange[-1]}_real.csv'),
                'EPS0.01':os.path.join(self.data_path, f'Confusion_matrix_Threshold_4_RHR_{rhrRange[0]}_{rhrRange[-1]}_epsilon_0.01.csv'), 
                'EPS0.02':os.path.join(self.data_path, f'Confusion_matrix_Threshold_4_RHR_{rhrRange[0]}_{rhrRange[-1]}_epsilon_0.02.csv'), 
                'EPS0.03':os.path.join(self.data_path, f'Confusion_matrix_Threshold_4_RHR_{rhrRange[0]}_{rhrRange[-1]}_epsilon_0.03.csv'), 
                'EPS0.04':os.path.join(self.data_path, f'Confusion_matrix_Threshold_4_RHR_{rhrRange[0]}_{rhrRange[-1]}_epsilon_0.04.csv'), 
                'EPS0.05':os.path.join(self.data_path, f'Confusion_matrix_Threshold_4_RHR_{rhrRange[0]}_{rhrRange[-1]}_epsilon_0.05.csv')
            }
        }

    self.threshold_df = pd.read_csv(self.data['ORG']['R'])
    self.threshold_df = self.threshold_df.loc[np.logical_and(self.threshold_df.nCol==2, self.threshold_df.RHR==100), 
                                    ['Freq', 'Sliding', 'THRESHOLD', 'F1 score','FNR']]
    self.threshold_df.sort_values(['Freq', 'Sliding', 'THRESHOLD'], inplace=True)
    self.threshold_df.reset_index(drop=True, inplace=True)
    self.threshold_df.THRESHOLD = [int(x*1000) for x in self.threshold_df.THRESHOLD]
   

    def __get_conf_int__(self, values, COI=0.95, verbose=0):
        l,h=st.t.interval(confidence=COI, 
                  df=len(values)-1, 
                  loc=np.mean(values),  
                  scale=st.sem(values))
        v = np.std(values)
        m = np.mean(values)
        
        if not np.isfinite(l):
            l=m
        if l<0:
            l=0
            
        if verbose>0:
            if l<0:
                print(values)
            print(f"m={m:.5f}, l={l:.5f}, h={h:.5f}, v={v:.5f}")
        
        return {
                'val': values,
                'mean': m,
                'std': v,
                'lowCOI': l,
                'highCOI': h,
                'dl': abs(m-l),
                'dh': abs(h-m)
               }

    def __initiateObj__(self, key, objDict):
        if key not in objDict:
            objDict[key] = {'F1':{}, 'FNR':{}}
            
    def __get_labels_color__(self, txt):
        if txt=='R':
            return 'gray'
        if txt=='S':
            return 'black'
        elif txt=='RS':
            return 'b'
        elif txt=='EPS0.01':
            return 'g'
        elif txt=='EPS0.02':
            return 'm'
        elif txt=='EPS0.03':
            return 'orange'
        elif txt=='EPS0.04':
            return 'r'
        elif txt=='EPS0.05':
            return 'brown'
        return ''
    
    def __get_labels__(self, txt):
        if txt=='R':
            return 'RD'
        elif txt=='S':
            return 'SD'
        elif txt=='RS':
            return 'RD & SD'
        elif txt=='EPS0.01':
            return '$\epsilon=0.01$'
        elif txt=='EPS0.02':
            return '$\epsilon=0.02$'
        elif txt=='EPS0.03':
            return '$\epsilon=0.03$'
        elif txt=='EPS0.04':
            return '$\epsilon=0.04$'
        elif txt=='EPS0.05':
            return '$\epsilon=0.05$'
        return ''
    
    def get_RHR_TH_dict(self, data_key, threshold_limit=None, conf_int=0.90,):
        dict_rhr = {
            'R':defaultdict(),
            'S':defaultdict(),
            'RS':defaultdict(), 
            'EPS0.01':defaultdict(), 
            'EPS0.02':defaultdict(), 
            'EPS0.03':defaultdict(), 
            'EPS0.04':defaultdict(), 
            'EPS0.05':defaultdict()
        }
        
        #print('dict_rhr')
        for k in dict_rhr:
            if k in self.data[data_key]:
                for rhr in rhrRange:
                    key = str(rhr)
                    #print(k, "->", key)
                    initiateObj(key, dict_rhr[k])
    
                    if threshold_limit is None:
                        df_tmp = pd.read_csv(self.data[data_key][k], usecols=['RHR','F1 score','FNR'])
                        df_tmp = df_tmp[df_tmp.RHR==rhr]
                        
                    elif threshold_limit > 0:
                        df_tmp = pd.read_csv(self.data[data_key][k], usecols=['RHR','F1 score','FNR', 'THRESHOLD'])
                        df_tmp = df_tmp[np.logical_and(df_tmp.RHR==rhr, df_tmp.THRESHOLD<=threshold_limit)]
                        df_tmp.THRESHOLD = [int(x*1000) for x in df_tmp.THRESHOLD]
    
                    if len(df_tmp['F1 score'].values)>0:
                        dict_rhr[k][key]['F1'] = get_conf_int(np.round(df_tmp['F1 score'].values, 3), conf_int)
    
                    if len(df_tmp['FNR'].values)>0:
                        dict_rhr[k][key]['FNR']= get_conf_int(np.round(df_tmp['FNR'].values, 3), conf_int)
            
        # Threshold vs performance
        
        dict_th = {
            'R':defaultdict(),
            'S':defaultdict(),
            'RS':defaultdict(), 
            'EPS0.01':defaultdict(), 
            'EPS0.02':defaultdict(), 
            'EPS0.03':defaultdict(), 
            'EPS0.04':defaultdict(), 
            'EPS0.05':defaultdict()
        }
        
        for k in dict_th:
            if k in self.data[data_key]:
                for threshold in thresholdRange:
                    key = str(round(threshold, 3))
                    initiateObj(key, dict_th[k])
    
                    df_tmp = pd.read_csv(self.data[data_key][k], usecols=['THRESHOLD','F1 score','FNR'])
                    df_tmp.THRESHOLD = [int(x*1000) for x in df_tmp.THRESHOLD]
                    df_tmp = df_tmp[df_tmp.THRESHOLD==threshold]
    
                    if len(df_tmp['F1 score'].values)>0:
                        dict_th[k][key]['F1'] = get_conf_int([round(x, 3) for x in df_tmp['F1 score'].values], conf_int)
    
                    if len(df_tmp['FNR'].values)>0:
                        dict_th[k][key]['FNR'] = get_conf_int([round(x, 3) for x in df_tmp['FNR'].values], conf_int)
                        
        return {"RHR":dict_rhr, "TH":dict_th}
    
    def get_user_data(self, user, synthetic=False):
        fn = f'{user}_hr.csv'
        if synthetic:
            fn = f'{user}_Syn_hr.csv'
    
        df_hr = pd.read_csv(os.path.join(self.data_path, f'{user}', fn), parse_dates=['datetime'], index_col=['datetime'])
        df_hr = df_hr[['heartrate']]
    
        fn = f'{user}_steps.csv'
        if synthetic:
            fn = f'{user}_Syn_steps.csv'
            
        df_steps = pd.read_csv(os.path.join(self.data_path, f'{user}', fn), parse_dates=['datetime'], index_col=['datetime'])
        df_steps = df_steps[['steps']]
        
        # Merge data
        df = df_hr.merge(df_steps, left_index=True, right_index=True)
        del[df_hr, df_steps]
        df = df.dropna()
        df.reset_index(inplace=True)
        
        return df[['datetime', 'steps', 'heartrate']]
    
    def get_user_plot_data(self, user, verbose=0):
        if verbose>0: 
            print('+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++')
            print()
            print(f'Processing for user: {user}')
    
        f = os.path.join(self.data_path, f'{user}', f'{user}_Syn_Summary.csv')
        summary_data = pd.read_csv(f)
        if verbose>0: print(summary_data.shape)
    
        df = self.get_user_data(user)
        if verbose>0: print(df.shape)
            
        pred_df = self.get_user_data(user, synthetic=True)
        if verbose>0: print(pred_df.shape)
    
        real_data = df.to_numpy()
        generated_data = pred_df.to_numpy()
    
        return [summary_data, real_data, generated_data]


    ############# Plotting functions
    
    def plot_window_sliding(self, plt, ax, label_pad=10, fs = 80, fsd = 20, lw = 4.0, mrkrSize = 30.0, mrkrFceClr = 'blue', 
                            mrkrEdgClr = 'blue', cpSize = 20.0, cpThik = 8.0, lineThik = 4.0, precision = 4, lineClr = 'blue', 
                            legend_size=45, title_prop = {'TEXT':"(A)", "Y_POS":1.0, "PAD":20, "LOC":'right'}):
    
        lc = ['r','g','b','m']
        
        def gettimes(t1, t2):
            def gettime(t):
                if t//60<=0:
                    return f"${t}$minutes"
                elif t//60==1:
                    return f"${t//60}$hour"
                else:
                    return f"${t//60}$hours"
                
            a = gettime(t1)
            b = gettime(t2)
            
            return f"$w$={a}, $s$={b}"
    
        for i, (f,s,msh,col) in enumerate([(60,30,"o",'r'),(60,60,"^",'g'),(120,60,"s",'b'),(120,120,"D",'m')]):
            tmp = self.threshold_df[np.logical_and(self.threshold_df.Freq==f, self.threshold_df.Sliding==s)]
            #print(f,s, tmp.THRESHOLD.unique())
            plt.plot(tmp.THRESHOLD, tmp['F1 score'], marker=msh, label=gettimes(f,s), lw=lineThik, color=lc[i], ms=mrkrSize, mfc=col, mec=col)
            
            mrkrFceClr = 'green'
            mrkrEdgClr = 'green'
            lineClr = 'green'
            legend_properties = {'weight':'normal', 'size':legend_size}
            heightF1 = 0.0
            
            plt.plot(tmp.THRESHOLD, tmp['FNR'], marker=msh,  lw=lineThik, color=lc[i], ms=mrkrSize, mfc=col, mec=col)
        
        plt.title(title_prop['TEXT'], y=title_prop['Y_POS'], pad = title_prop['PAD'], loc = title_prop['LOC'], 
                  fontsize = title_prop['FS'], fontweight=title_prop['FW'], fontfamily='Times New Roman')
        plt.text(5, .92, '$F_1$ score', fontsize = fs-fsd-10, fontweight='normal', fontfamily='Times New Roman', backgroundcolor='yellow')
        plt.text(5, .04, 'FNR', fontsize = fs-fsd-10, fontweight='normal', fontfamily='Times New Roman', backgroundcolor='yellow')
        plt.text(19.5, -0.16, '$x10^{-3}$', fontsize = fs/2, fontweight='normal', fontfamily='Times New Roman')
        
        yt = np.arange(0, 1.1, 0.2)
        plt.xticks(ticks=tmp.THRESHOLD.unique()[::2], labels=tmp.THRESHOLD.unique()[::2], rotation=0, size=fs-fsd, )
        plt.ylim(-0.1,1.1)
        plt.yticks(ticks=yt, labels=[str(x)[:3] for x in yt], weight='normal', size=fs-fsd)
        plt.xlabel('Threshold ($\Delta$) on Hellinger distance', fontsize=fs, labelpad=label_pad)
        plt.ylabel('Performance metrics', fontsize=fs, labelpad=label_pad )
        plt.legend(loc="center left", prop=legend_properties)
        
    def plot_all_user(self, plt, ax, fig, perf, obj_dict, data, use_selected_index=False, use_selected_index_color=False, inset_box_width=None,
                      inset_box_height=None, inset_loc=3, inset_borderpad=1, inset_y_lim=None, inset_precision_offset=0, inset_border_lw=6.0,
                      inset_cnt=5, inset_fs=20, show_x_ticks=True, show_y_ticks=True, show_legend=False, xlabel='', xrotation=0, 
                      show_ylabel=False, label_pad=10, fw = 'normal', fs = 80, fsd = 20, lw = 4.0, mrkrSize = 3.0, mrkrFceClr = 'blue', 
                      mrkrEdgClr = 'blue', cpSize = 3.0, cpThik = 3.0, lineThik = 4.0, precision = 4, lineClr = 'blue', legend_size=45,
                      legend_pos="center right", legend_cols=1, xtr_param=(-60, 30), title_prop = None):
        show_inset = True
        axes = None
        
        if inset_box_width is None or inset_box_height is None:
            show_inset = False
        
        if show_inset:
            #axes = plt.axes(inset_box_pos)
            axes = inset_axes(ax, width=f"{inset_box_width}%", height=f"{inset_box_height}%", loc=inset_loc, borderpad=inset_borderpad)
    
        xfr = xtr_param[0]
        for key in obj_dict:
            if key not in data:
                #print(f"key={key} not in data")
                continue
    
            #print(key)
    
            try:
                #transX = ax.transData + ScaledTranslation(xfr/72, 0, fig.dpi_scale_trans)
                transX = ax.transAxes + ScaledTranslation(xfr/72, 0, fig.dpi_scale_trans)
                
                if show_inset:
                    #trans_inset_X = axes.transData + ScaledTranslation(xfr/72, 0, fig.dpi_scale_trans)
                    trans_inset_X = axes.transAxes + ScaledTranslation(xfr/72, 0, fig.dpi_scale_trans)
            except Exception as e:
                print(f"Error on transX={e}")
            finally:
                transX=None
                trans_inset_X=None
                
            xfr +=xtr_param[1]
            
            obj = obj_dict[key]
            list_key = list(obj.keys())
            list_mean = [obj[k][perf]['mean'] for k in obj.keys()]
            list_COI = [obj[k][perf]['dl'] for k in obj.keys()]
    
            sel_index = [i for i in range(len(list_key)) if i%2==0]
            if use_selected_index:
                sel_index = [0,2,3,4,6,8,10,12,14]
    
            colors = [get_labels_color(key)]*len(sel_index)
            lbl = get_labels(key)
            lk = [list_key[i] for i in sel_index]
            #print(key, lbl, lk)
    
            # Make red color on 3rd index
            if use_selected_index and use_selected_index_color:
                for i,d in enumerate(sel_index):
                    if d==3:
                        if key=='R':
                            colors[i] = 'red'
                        elif key=='S':
                            colors[i] = 'm'
                        elif key=='RS':
                            colors[i] = 'black'
    
            for cnt, i in enumerate(sel_index):
                if cnt==0:
                    if show_legend:
                        ax.errorbar(x=list_key[i], 
                                     y=list_mean[i], 
                                     yerr = list_COI[i],
                                     linestyle='', marker='o', ms=mrkrSize, mfc=colors[cnt], mec=colors[cnt], 
                                     capsize=cpSize, capthick=cpThik, transform=transX,
                                     elinewidth=lineThik, ecolor=colors[cnt], label=lbl)
                    else:
                        ax.errorbar(x=list_key[i], 
                                     y=list_mean[i], 
                                     yerr = list_COI[i],
                                     linestyle='', marker='o', ms=mrkrSize, mfc=colors[cnt], mec=colors[cnt], 
                                     capsize=cpSize, capthick=cpThik, transform=transX,
                                     elinewidth=lineThik, ecolor=colors[cnt])
    
                else:
    
                    ax.errorbar(x=list_key[i], 
                                     y=list_mean[i], 
                                     yerr = list_COI[i],
                                     linestyle='', marker='o', ms=mrkrSize, mfc=colors[cnt], mec=colors[cnt], 
                                     capsize=cpSize, capthick=cpThik, transform=transX,
                                     elinewidth=lineThik, ecolor=colors[cnt])
    
                if show_inset and cnt<inset_cnt:
                    #print(f"({r},{c}) {key}: m={list_mean[i]:.5f}, COI={list_COI[i]:.5f}, [{(list_mean[i]-list_COI[i]):.5f}, {(list_mean[i]+list_COI[i]):.5f}]")
                    axes.errorbar(x=list_key[i], 
                                     y=list_mean[i], 
                                     yerr = list_COI[i],
                                     linestyle='', marker='o', ms=mrkrSize, mfc=colors[cnt], mec=colors[cnt], 
                                     capsize=cpSize, capthick=cpThik, transform=trans_inset_X,
                                     elinewidth=lineThik, ecolor=colors[cnt])
                    
                    
        if show_x_ticks:
            ax.set_xticklabels(lk, rotation=xrotation, size=fs-fsd, weight=fw)
            
            if use_selected_index:
                ax.text(7.4, -0.31, '$x10^{-3}$', fontsize = fs/2, fontweight='normal', fontfamily='Times New Roman')
        
        else:
            ax.set_xticklabels([])
    
        ax.set_ylim(-0.1,1.1)
        
        if show_y_ticks:
            #yt = [str(x)[:precision] for x in ax.get_yticks()]
            yv = np.arange(0, 1.1, 0.5)
            yt = [str(x)[:precision+1] for x in yv]
            #ax.set_yticklabels(yt, weight=fw, size=fs-fsd)
            ax.set_yticks(yv, yt, weight=fw, size=fs-fsd)
        else:
            ax.set_yticklabels([])
    
        if show_inset:
            axes.set_xticklabels([])
            
            if inset_y_lim is not None:
                axes.set_ylim(inset_y_lim)
                
            axes.yaxis.tick_right()
            
            #yt = [x if idx%2==0 else '' for idx, x in enumerate([str(x)[:precision+inset_precision_offset] for x in axes.get_yticks()])]
            ytt = [str(x)[:precision+inset_precision_offset] for x in axes.get_yticks() if x>=0]
            #print(ytt)
            yt = [x if (len(ytt)>6 and idx%2==0) or (len(ytt)<=6 and idx%2==1) else '' for idx, x in enumerate(ytt)]
            # print(yt)
                
            axes.set_yticklabels(yt, weight=fw, size=inset_fs)
           
            axes.grid('on',alpha = 0.9)
            plt.setp(axes.spines.values(), lw=inset_border_lw, color='black', alpha=1.0)
            axes.tick_params(bottom=False)
    
        #['F1', 'FNR']
        if show_x_ticks:
            ax.set_xlabel(xlabel, fontsize=fs, fontweight=fw, labelpad=label_pad)
        
        if show_y_ticks:
            if perf=='F1' and show_ylabel:
                ax.set_ylabel('F$_1$ score', fontsize=fs, fontweight=fw, labelpad=label_pad)
            elif perf=='FNR' and show_ylabel:
                ax.set_ylabel('FNR', fontsize=fs, fontweight=fw, labelpad=label_pad)
        
        #ax.grid('on',alpha = 1.0)
    
        if show_legend:
            ax.legend(loc=legend_pos, ncol=legend_cols, prop={'size':legend_size})
            
        ax.set_title(title_prop['TEXT'], y=title_prop['Y_POS'], pad = title_prop['PAD'], zorder=50, 
                     loc = title_prop['LOC'], fontsize = title_prop['FS'], fontweight=title_prop['FW'], 
                     fontfamily='Times New Roman')
    
    def plot_2Cols(self, data, write_file):
        def get_dateTime(dt):
            mn = f'0{dt.month}'[-2:]
            day = f'0{dt.day}'[-2:]
            hour = f'0{dt.hour}'[-2:]
            minute = f'0{dt.minute}'[-2:]
    
            return dt.strftime('%b-%d')#f'{mn}-{day}'# {hour}:{minute}'
        
        data.reset_index(drop=True, inplace=True)
        a = [i for i in range(0, data.shape[0], 100)]
        u = data.User.unique()[0]
        
        #fig, ax = plt.subplots(1+ncol,1,figsize=(35,30), constrained_layout=True, sharex=True) 
        fig, ax = plt.subplots(2,1,figsize=(65,40), dpi=300, ) 
        
        plt.subplots_adjust(hspace=.3)
        
        fs = 100
        fsd = 10
        lw = 10.0
        title_pad = 20.0
        
        axi = 0
        ax[axi].plot(data.index, data.hellingerDistance, color='blue', linewidth=lw)
        #ax[axi].plot(data[data.Status==1].index, data.hellingerDistance[data.Status==1], color='red', linewidth=lw)
        yt = [str(round(x,2)) for x in ax[axi].get_yticks()]
        ax[axi].set_ylabel('Hellinger distance', fontsize=fs, fontweight='normal')
        ax[axi].set_xlim(min(a), max(a))
        ax[axi].set_xticklabels([]) #tick_params(axis="x", labelsize=20)
        ax[axi].set_yticklabels(yt, weight='normal', size=fs-fsd) #tick_params(axis="y", labelsize=20)
        ax[axi].set_title(f'{u}', fontsize=fs, fontweight='bold', pad=title_pad)
        ax[axi].grid(True, color = 'darkgreen')
        ax[axi].axhline(y = 0.008, color = 'm', linestyle = '-', linewidth=lw-0.5)
        ax[axi].text(10.0, 0.16, "(a)", fontweight='bold', fontsize=fs+10)
        
        axi+=1
        ax[axi].plot(data.index, data.avgRHR, color='blue', linewidth=lw)
        ax[axi].plot(data[data.Status==1].index, data.avgRHR[data.Status==1], color='red', linewidth=lw)
        yt = [int(x) for x in ax[axi].get_yticks()]
        ax[axi].set_ylabel('RHR', fontsize=fs, fontweight='normal')
        ax[axi].set_xlim(min(a), max(a))
        ax[axi].set_xticks(a, [get_dateTime(d) for d in data.datetime[a]], rotation=90, weight='normal', size=fs-fsd)
        ax[axi].set_yticklabels(yt, weight='normal', size=fs-fsd) #tick_params(axis="y", labelsize=20)
        ax[axi].set_title('')
        ax[axi].grid(True, color = 'darkgreen')
        ax[axi].axhline(y = 100, color = 'm', linestyle = '-', linewidth=lw-0.5)
        ax[axi].text(10.0, 130, "(b)", fontweight='bold', fontsize=fs+10)
        
        for r in range(2):
            plt.setp(ax[r].spines.values(), lw=10, color='black', alpha=1.0)
            
        if len(write_file)>0 and os.path.exists(write_file):
            plt.savefig(os.path.join(write_file, f'{u}_HD.png'), dpi=300, format='png', transparent=False)
            plt.close()
        else:
            plt.show()
    
    def plot_multiple_users(self, user_list, verbose=0):
        fig = plt.figure(figsize=(60,25), dpi=300, clear=True)
        gs = gridspec.GridSpec(1, 7, figure=fig, top=0.98, left=0.06, bottom=0.1, right=0.995, hspace=0.2, wspace=0.5)
        gs0 = gridspec.GridSpecFromSubplotSpec(2, 2, subplot_spec=gs[:3], hspace=0.1, wspace=0.3)
        gs1 = gridspec.GridSpecFromSubplotSpec(2, 1, subplot_spec=gs[3:], hspace=0.05, wspace=0.03)
    
        axiss = []
        
        fs = 70
        fsd = 10
        ms = 300
        
        plt.rcParams['font.size'] = fs
        plt.rcParams['ytick.labelsize'] = fs-fsd
        plt.rcParams['xtick.labelsize'] = fs-fsd
        plt.rcParams['font.family'] = 'Times New Roman'
        plt.rcParams['font.weight'] = 'normal'
        mpl.rcParams['axes.linewidth'] = 5.0
                
        r=0
        title = [('(a)','(b)','(c)'),('(d)','(e)','(f)')]
        title_font_size=fs+10
        title_pad=-50
        title_zorder=200
        
        for u in user_list:
            summary_data, real_data, generated_data = get_user_plot_data(u, verbose)
            
            ## PDF difference between real and synthetic data at different kinds of heart rate against Epochs
            ax = plt.subplot(gs0[r, 0])
            axiss.append(ax)
            ax.text(-12000, 0.15, u, fontweight='bold', fontsize=fs+20, rotation=90)
            
            if u==user_list[-1]:
                plt.plot(summary_data.epocs, summary_data.RHR_distance, color='r', label=f'RHR', linewidth=5)
                plt.plot(summary_data.epocs, summary_data.AHR_distance, color='b', label=f'AHR', linewidth=5)
                plt.plot(summary_data.epocs, summary_data.PR_distance, color='g', label=f'OHR', linewidth=5)
                plt.legend(prop={'weight':'normal', "size": fs-fsd})
                plt.tick_params(direction='out', length=20, width=6)
            else:
                plt.plot(summary_data.epocs, summary_data.RHR_distance, color='r')
                plt.plot(summary_data.epocs, summary_data.AHR_distance, color='b')
                plt.plot(summary_data.epocs, summary_data.PR_distance, color='g')
                ax.set_xticklabels([])
                plt.tick_params(axis='y',direction='out', length=20, width=6)
            
            plt.title(title[r][0], loc='center', y=0.98, pad=title_pad, fontweight='bold', fontsize=title_font_size, zorder=title_zorder)
            plt.xlim(0, 33500)
            plt.ylim(0, 0.6)
            plt.ylabel(f'PDF Difference', fontsize=fs)
            if u==user_list[-1]: 
                plt.xlabel('Epochs', fontsize=fs)
            
            ## PDF between real and synthetic data 
            real_rhr = list(real_data[np.where(real_data[:,1]==0), 2][0])
            fake_rhr = list(generated_data[np.where(generated_data[:,1]==0), 2][0])
    
            ax = plt.subplot(gs0[r, 1])
            axiss.append(ax)
    
            if u==user_list[-1]:
                sns.distplot(real_rhr, hist=False, color='red', ax=ax, label='RD', kde_kws=dict(linewidth=5))
                sns.distplot(fake_rhr, hist=False, color='blue', ax=ax, label='SD', kde_kws=dict(linewidth=5))
                plt.legend(prop={'weight':'normal', "size": fs-fsd})
                plt.tick_params(direction='out', length=20, width=6)
            else:
                sns.distplot(real_rhr, hist=False, color='red', ax=ax, kde_kws=dict(linewidth=5))
                sns.distplot(fake_rhr, hist=False, color='blue', ax=ax, kde_kws=dict(linewidth=5))
                ax.set_xticklabels([])
                plt.tick_params(axis='y',direction='out', length=20, width=6)
                
            plt.title(title[r][1], loc='center', y=0.98, pad=title_pad, fontweight='bold', fontsize=title_font_size, zorder=title_zorder)
            plt.ylim(0, 0.04)
            plt.ylabel('PDF', fontsize=fs)
            if u==user_list[-1]: 
                plt.xlabel('RHR', fontsize=fs)
        
            ## Scatter plot of real and synthetic data
            ax = plt.subplot(gs1[r, :])
            axiss.append(ax)
            
            if u==user_list[-1]:
                plt.scatter(real_data[:, 1], real_data[:, 2], color='r', label='RD', s=ms)
                plt.scatter(generated_data[:, 1], generated_data[:, 2], color='b', label='SD', s=ms)
                plt.legend(prop={'weight':'normal', "size": fs-fsd})
                plt.tick_params(direction='out', length=20, width=6)
            else:
                plt.scatter(real_data[:, 1], real_data[:, 2], color='r', s=ms)
                plt.scatter(generated_data[:, 1], generated_data[:, 2], color='b', s=ms)
                ax.set_xticklabels([])
                plt.tick_params(axis='y',direction='out', length=20, width=6)
                
            plt.title(title[r][2], loc='center', y=0.98, pad=title_pad, fontweight='bold', fontsize=title_font_size, zorder=title_zorder)
            plt.xlim(-2, 180)
            plt.ylabel('Heart rate', fontsize=fs+10)
            if u==user_list[-1]: 
                plt.xlabel('Steps', fontsize=fs+10)
            
            r+=1
    
                
        for axes in axiss:
            plt.setp(axes.spines.values(), lw=6, color='black', alpha=1.0)
    
        plt.savefig(os.path.join('./Figure/', 'WGAN_results.png'), dpi=300, format='png', transparent=False)
        plt.close()



