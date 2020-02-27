import schedule
import time

def job():
    print("I am doing this job!")


schedule.every().monday.at("14:00").do(job)
schedule.every().tuesday.at("14:00").do(job)
schedule.every().wednesday.at("14:00").do(job)
schedule.every().thursday.at("14:00").do(job)
schedule.every().friday.at("14:00").do(job)

while True:
    schedule.run_pending()
    time.sleep(1)