import pandas as pd
def fisher_score_list(data, target, topn ):
     
    items = list(data)

    num_classes = len(set(target))

    fisher_score = []

    grouped = data.groupby(target, as_index=False)

    n = [len(data[target == k+1]) for k in range(num_classes)]
    
    
    for i in items:
        temp = grouped[i].agg({str(i)+'_mean': 'mean',
                               str(i)+'_std': 'std'})

        numerator = 0
        denominator = 0

        u_i = data[i].mean()

        for k in range(num_classes):
            n_k = n[k]
            u_ik = temp.iloc[k, :][str(i)+'_mean']
            p_ik = temp.iloc[k, :][str(i)+'_std']

            numerator += n_k*(u_ik-u_i)**2
            denominator += n_k*p_ik**2

        fisher_score.append(numerator/denominator)

    fisher_score = pd.DataFrame(fisher_score)
    fisher_score['Col'] = items = list(data)
    fisher_score = fisher_score.sort_values([0], ascending=False)
    fisher_score = fisher_score[0:topn]
    fisher_score_l = fisher_score['Col'].iloc[0:topn]
    f_score_l = list(fisher_score_l)
    return f_score_l