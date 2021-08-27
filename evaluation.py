# Import Libraries


# Accuracy Measure

def calculate_accuracy(y_ts,yh):
    
    N = len(y_ts)
    hits = 0
    
    for i in range(N):
        if(y_ts[i] == yh[i]):
            hits = hits + 1
    accuracy = hits/N
    return accuracy

def calculate_accuracy2(y_ts,yh):
    N = len(y_ts)
    hits = 0
    for i in range(N):
        if(y_ts[i] > 0 and yh[i] > 0):
            hits = hits + 1
        elif(y_ts[i] < 0 and yh[i] < 0):
            hits = hits + 1
    
    accuracy = hits/N
    return accuracy          