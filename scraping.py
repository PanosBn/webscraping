import pandas as pd
import json, requests, webbrowser, re, math, string
import datetime
from bs4 import BeautifulSoup
import csv


def get_internet_archive():
    # lista me paradeigmata apo kathe istoselida
    websites = ['bbc.com/news', 'guardian.co.uk', 'nytimes.com', 'independent.co.uk', 'https://www.thetimes.co.uk/']

    # Me ta pandas.date_range dimiourgoume ena sunolo apo imerominies gia kathe mera apo to 1998 - 2017
    start_date = datetime.date(2003, 1, 1)
    end_date = datetime.date(2017, 12, 30)

    timestamps = []
    daterange = pd.date_range(start_date, end_date)
    for single_date in daterange:
        dateString = str(single_date.year) + str(single_date.month) + str(single_date.day) + str(120000)
        # print(dateString)
        timestamps.append(dateString)
    http_responses = 0
    snapshots = dict()

    for x in range(0, len(websites)):
        for timestamp in timestamps:
            try:
                req = 'http://archive.org/wayback/available?url=' + websites[x] + '/&timestamp=' + timestamp
                # source_website = re.findall("((http:|https:)//[^ \<]*[^ \<\.])", line)
                print(type(req))
                print(req)
                r = requests.get(req, verify=False)
                print(r)
                if r:
                    http_responses = http_responses + 1
                print(http_responses)
                data = r.json()
                availabillity = data['archived_snapshots']['closest']['available']

                if (availabillity == True):
                    # print("Page exists")
                    url = data['archived_snapshots']['closest']['url']
                    response = requests.get(url)
                    html = response.text
                    soup = BeautifulSoup(html, "html.parser")
                    for script in soup.findAll('script'):
                        script.extract()
                    [s.extract() for s in soup('style')]  # afairesi css kwdika apo selides pou periexoun (Note: Sumvainei mono sti selida twn NYT)

                    text = soup.body.getText()  # to text pou exei apomeinei
                    stripped_text = text.splitlines()  # spaei me newline

                    stripped_text = list(filter(None, stripped_text))
                    final_list = []
                    for text in stripped_text:
                        text.strip()
                        regx = re.compile(r"\s+")  # merikes ergasies me regex gia na afairethoun leading whitespaces, na sumbitxthoun polla spaces se ena ktl
                        text = regx.sub(repl=" ", string=text)
                        text.strip(" ")
                        regx = re.compile(r"([^\w\s]+)|([_-]+)")
                        text = regx.sub(repl=" ", string=text)
                        sentence_length = (len(text.split()))
                        if (sentence_length > 3):  # Epilegetai na diatirithoun oses eidiseis exoun panw apo 3 lekseis
                            final_list.append(text)
                    # for text in final_list:
                    #     print(text)
                    # snapshots[websites[x] + " :" + timestamp] = final_list
                    snapshots[req] = final_list
                else:
                    print("'Tis not here")
            except Exception:
                print("ERROR" + " " + req)

    return snapshots


if __name__ == '__main__':

    csv_file = "archivedNews.csv"
    snapshots = get_internet_archive()
    with open('archivedNews.csv', 'w') as f:
        fieldnames = ['Timestamp', 'Headlines']
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        data = [dict(zip(fieldnames, [k, v])) for k, v in snapshots.items()]
        writer.writerows(data)

        # for line in snapshots:
    #     print(line)
    #     print(snapshots[line])
