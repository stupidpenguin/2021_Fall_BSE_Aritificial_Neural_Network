import time

def Get_Current_time_and_date():
    localtime = time.localtime()
    result = time.strftime("%Y-%m-%d_%I-%M-%S_%p", localtime)
    time.sleep(1)
    return result

def Get_Current_time():
    localtime = time.localtime()
    result = time.strftime("%I:%M:%S %p", localtime)
    time.sleep(1)
    return result