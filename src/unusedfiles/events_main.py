


def classify(steps, met, dura, th1, th2, th3):
    """
    Extract events and classify:
    Long walks:  th2< steps < th3, duration > 30 mins,
    Short walk: 5-10 min
    Medium: walk 10-30 min
    workout Run: steps > th3,  dura>15
    short run: steps > th3,  dura<15
    Stand/sit: steps=0 AND MET > th1,
    Sleep: steps=0 AND MET < th1

    CODE :
    Sleep: 0
    Stand/sit: 1
    Short walk: 2
    medium walk: 3
    long walk:  4
    short run: 5
    workout run: 6
    NONE: -1
    """

    if steps==0:
        if met < th1:
            return 0
        else:
            return 1
    elif th2 < steps <th3:
        if 5 < dura < 10:
            return 2
        elif 10 < dura < 30:
            return 3
        elif dura > 30:
            return 4
        else:
            return -1
    elif steps >=th3:
        if dura < 15:
            return 5
        else:
            return 6
    else:
        return -1

type=classify()

"""
            
event is a triple (duration, type,  start_time)
        
user    event1, event2, event3 ...

1535    (15,3,8:00)-->distribution 

"""



