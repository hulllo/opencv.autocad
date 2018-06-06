 


def count_same(lists):  
    count_times = [] 
    samecount = 1 #自身数量为1个
    samecounts = []
    values = []
    for n,i in enumerate(lists) :
        if len(values) == 0:
                 values.append(i)
        elif i != values[-1]:
            values.append(i)
            

        if n + 1 < len(lists):
            
            if i == lists[n+1]: 
                samecount = samecount + 1 

            else:
                samecounts.append(samecount)
                samecount = 1
    samecounts.append(samecount) 
    return(values,samecounts)         


if __name__ == '__main__':
    lists = [1,1,1,1,0,0,0,1,1,1,0,1,1,1,1,0]
    # lists = [1,0]

    values,samecounts = count_same(lists)
    print(values)    
    print(samecounts)
