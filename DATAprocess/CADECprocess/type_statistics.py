import json
import pathlib

class Statistics:
    def __init__(self,data_config):
        self.data_config = data_config

    def freq(self):
        p = pathlib.Path(self.data_config['ann_filePath'])
        file_names = list(p.glob('*'))
        data = []
        gap_freq={}
        total_freq={}
        for name in file_names:
            n = str(name)
            label = {}
            with open(n, 'r') as f:
                for line in f:
                    if '#' in line:
                        continue
                    line = line.replace('\n', '')
                    ls = line.split('\t')
                    space_pos = ls[1].find(' ')
                    type_txt = ls[1][:space_pos]
                    if type_txt not in gap_freq:
                        gap_freq[type_txt]=0
                    if type_txt not in total_freq:
                        total_freq[type_txt]=1
                    else:
                        total_freq[type_txt]+=1
                    index_range = ls[1][(space_pos + 1):]
                    if ';' in index_range:
                        gap_freq[type_txt]+=1
        return gap_freq,total_freq

    def percent(self,gap_freq,total_freq):
        gap_percent={}
        for key in total_freq:
            gap_percent[key] = gap_freq[key]/total_freq[key]
        return gap_percent

if __name__ == "__main__":
    data_config={'ann_filePath':'/datastore/liu121/nosqldb2/cadec/original',
                 'stat_filePath':'/datastore/liu121/nosqldb2/cadec/stat.json'}
    s = Statistics(data_config)
    gap_freq,total_freq = s.freq()
    gap_percent = s.percent(gap_freq,total_freq)
    statistics={}
    for key in total_freq:
        statistics[key]={'total':total_freq[key],'gap':gap_freq[key],'ratio':gap_percent[key]}
    with open(data_config['stat_filePath'],'w+') as f:
        json.dump(statistics, f, indent=4, sort_keys=False)