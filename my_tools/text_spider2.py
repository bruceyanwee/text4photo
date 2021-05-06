from typing import Text
from lxml import etree
import requests
import random
import time
import pandas as pd
# 反爬虫用到的
proxy_list = [
    '120.83.105.119:9999',
]
# 请求网页传的参数
# 收集到的常用Header
my_headers = [
    "Mozilla/5.0 (Windows NT 6.3; WOW64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/39.0.2171.95 Safari/537.36",
    "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_9_2) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/35.0.1916.153 Safari/537.36",
    "Mozilla/5.0 (Windows NT 6.1; WOW64; rv:30.0) Gecko/20100101 Firefox/30.0",
    "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_9_2) AppleWebKit/537.75.14 (KHTML, like Gecko) Version/7.0.3 Safari/537.75.14",
    "Mozilla/5.0 (compatible; MSIE 10.0; Windows NT 6.2; Win64; x64; Trident/6.0)",
    'Mozilla/5.0 (Windows; U; Windows NT 5.1; it; rv:1.8.1.11) Gecko/20071127 Firefox/2.0.0.11',
    'Opera/9.25 (Windows NT 5.1; U; en)',
    'Mozilla/4.0 (compatible; MSIE 6.0; Windows NT 5.1; SV1; .NET CLR 1.1.4322; .NET CLR 2.0.50727)',
    'Mozilla/5.0 (compatible; Konqueror/3.5; Linux) KHTML/3.5.5 (like Gecko) (Kubuntu)',
    'Mozilla/5.0 (X11; U; Linux i686; en-US; rv:1.8.0.12) Gecko/20070731 Ubuntu/dapper-security Firefox/1.5.0.12',
    'Lynx/2.8.5rel.1 libwww-FM/2.14 SSL-MM/1.4.1 GNUTLS/1.2.9',
    "Mozilla/5.0 (X11; Linux i686) AppleWebKit/535.7 (KHTML, like Gecko) Ubuntu/11.04 Chromium/16.0.912.77 Chrome/16.0.912.77 Safari/535.7",
    "Mozilla/5.0 (X11; Ubuntu; Linux i686; rv:10.0) Gecko/20100101 Firefox/10.0 "
]
# 获取 'https://www.juzikong.com/' 首页的HTML
url_test_ip = "http://httpbin.org/ip"
url_yipinjuzi = 'http://www.yipinjuzi.com/'
header = {'User_Agent':random.choice(my_headers)}
proxy = {'https':random.choice(proxy_list)}
def return_html(url):
    r = requests.get(url,headers = header)
    r.encoding='utf-8'
    html = etree.HTML(r.text)
    return html
if __name__ == "__main__":
    tag_list = {
        "negtive":[
            'shangganjuzi/',
            'xinqingbuhao/',
            'wunaijuzi/',
            'shiwangjuzi/',
            ],
        "positive":[
            'jijixiangshang/',
            'mubiaojuzi/',
            'gulirende/',
            'zhengnengliangde/']
    }  
    tag_name_dict = {
        'shangganjuzi/':'伤感',
        'xinqingbuhao/':'伤心',
        'wunaijuzi/':'无奈',
        'shiwangjuzi/':'失望',
        'jijixiangshang/':'乐观',
        'mubiaojuzi/':'励志',
        'gulirende/':'鼓励',
        'zhengnengliangde/':'正能量'
    }
    # 爬取一些消极极性的文案，senti=neg,tag=对应的tag
    for senti_p in ['negtive','positive']:
        for tag_name in tag_list[senti_p]:
            print("downloading..."+tag_name_dict[tag_name])
            res_texts = []
            html = return_html(url_yipinjuzi+tag_name)
            page_urls = html.xpath("//div[@class='Post']//div[@class='PostHead2']//a[2]/@href")
            page_titles = html.xpath("//div[@class='Post']//div[@class='PostHead2']//a[2]/text()")
            # page_list = ['/xinsuijuzi/15671.html',
            #  '/xintengjuzi/58.html',
            #  '/shangxin/59.html',
            #  '/xintengjuzi/60.html',
            #  '/xinqingbuhao/23582.html']
            for u,t in zip(page_urls,page_titles):
                # 有些页面是广告页面，根据标题中一些字来排除
                print("downloading..."+tag_name_dict[tag_name]+"page ing...")
                if '粉丝' in t or '刷' in t:
                    continue
                html_page = return_html(url_yipinjuzi+u[1:])
                txt_list = html_page.xpath("//div[@class='PostContent']/p/text()")
                texts = []
                for t in txt_list:
                    idx = t.find('、')
                    if idx != -1:
                        texts.append(t[idx+1:])
                res_texts.extend(texts)
            res_tag_names = [tag_name_dict[tag_name]]*len(res_texts)
            res_senti = [senti_p]*len(res_texts)
            df = pd.DataFrame({
                'text':res_texts,
                'tag':res_tag_names,
                'senti_polarity':res_senti
            })
            file_name = 'yipin_tag_'+tag_name_dict[tag_name]+'.csv'
            df.to_csv('../res_text/'+file_name)
            
            
