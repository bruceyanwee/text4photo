from lxml import etree
import requests
import random
from anti_spider import *
import time
import pandas as pd
# 获取 'https://www.juzikong.com/' 首页的HTML
url_test_ip = "http://httpbin.org/ip"
url_jinjugu = 'https://www.jinjugu.com'
def return_html(url):
    r = requests.get(url,headers = header)
    r.encoding='utf-8'
    html = etree.HTML(r.text)
    return html
header = {'User_Agent':random.choice(my_headers)}
proxy = {'https':random.choice(proxy_list)}
html = return_html(url_jinjugu+'/taglist/')
# 获取各种类别的名字和子路径
tag_paths,tag_names = html.xpath("//*[@id='snsBox']/div[1]//a/@href"),html.xpath("//*[@id='snsBox']/div[1]//a/text()")
for tag_path,tag_name in list(zip(tag_paths,tag_names))[100:200]:
    # 防止被封，延迟一点时间
    print("downloading..."+tag_name)
    time.sleep(random.random()*3)
    url_tag_page = url_jinjugu+tag_path
    html = return_html(url_tag_page)
    # 存放爬取的数据
    text_list = []
    tag_list = []
    author_list = []
    book_list = []
    #
    txt_divs = html.xpath("//div[@class='item statistic_item']")
    for t_div in txt_divs:
        text_list.append(t_div.xpath("normalize-space(./a/text())"))
        author_list.append(t_div.xpath("normalize-space(./div[1]/a[1]/text())"))
        book_list.append(t_div.xpath("normalize-space(./div[1]/a[2]/text())"))
    # 如果有多页，还需要继续爬取
    remain_pages_paths = set(html.xpath("//div[@class='pager']//a[@target]/@href"))
    # 爬取后面的页码
    for next_page_path in remain_pages_paths:
        time.sleep(random.random()*3)
        html_next = return_html(url_jinjugu+next_page_path)
        txt_divs = html_next.xpath("//div[@class='item statistic_item']")
        for t_div in txt_divs:
            text_list.append(t_div.xpath("normalize-space(./a/text())"))
            author_list.append(t_div.xpath("normalize-space(./div[1]/a[1]/text())"))
            book_list.append(t_div.xpath("normalize-space(./div[1]/a[2]/text())"))
    # 保存一下该 tag的文案
    tag_list = [tag_name]*len(text_list)
    df = pd.DataFrame({'text' : text_list, 'tag' : tag_list, 'author' : author_list,'from_book':book_list})
    file_name = 'tag_'+tag_name+'.csv'
    df.to_csv('../res_text/'+file_name)

