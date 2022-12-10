import sys
import requests as rq
from bs4 import BeautifulSoup as bs
import pandas as pd
import re
from re import findall
import urllib.parse
from pandas import RangeIndex


def make_hyperlink(value):
    url = "https://custom.url/{}"
    return '=HYPERLINK("%s", "%s")' % (url.format(value), value)


if len(sys.argv) < 3:
    print("Usage: python3 scrape.py <date> <xslx>")
    exit(1)

desired_date = sys.argv[1]
outpath = sys.argv[2]

url = "https://discoveratlanta.com/events/all/"

## Your code here

pageData = rq.get(url)
# print(pageData.text);

soup = bs(pageData.content, "html.parser")
links = soup.find_all("article")

link_date = soup.find_all("article", {"data-eventdates": re.compile(str(sys.argv[1]))})

eventUrls = []
maxcharsize = 0
for i in link_date:
    wholeLink = i.find("a")
    maxcharsize = max(maxcharsize, len(wholeLink["href"]))
    eventUrls.append(
        '=HYPERLINK("' + wholeLink["href"] + '", "' + wholeLink["href"] + '")'
    )

eventName = []
maxchartitle = 0
for i in link_date:
    en = i.find("h4")
    maxchartitle = max(maxchartitle, len(en.text))
    eventName.append(en.text)

d = {"title": eventName, "link": eventUrls}

df = pd.DataFrame(data=d, columns=["title", "link"])

# , engine="xlsxwriter"
writer = pd.ExcelWriter(outpath, engine="xlsxwriter")
df.to_excel(writer, index=False)
for column in df:
    column_width = max(df[column].astype(str).map(len).max(), len(column))
    col_idx = df.columns.get_loc(column)
    writer.sheets["Sheet1"].set_column(col_idx, col_idx, column_width)
writer.save()
# df.to_excel(str(sys.argv[2]),sheet_name='events', index= False)

# print(df)

# df.to_excel(outpath, sheet_name='Sheet_name_1')
