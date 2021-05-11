from lxml import etree
import pandas as pd
import sys
import os
sys.path.append('..')
parser = etree.HTMLParser(encoding="utf-8")
html = etree.parse("./res_text/travel.html", parser=parser)
l_text = html.xpath("//blockquote/text()")
l_tag = ['旅游']*len(l_text)
df = pd.DataFrame({'text':l_text,'tag':l_tag})
outname = '旅游.csv'
outdir = './res_text'
full_name = os.path.join(outdir,outname)
df.to_csv(full_name)