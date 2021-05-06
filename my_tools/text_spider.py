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
url_jinjugu = 'https://www.jinjugu.com'
header = {'User_Agent':random.choice(my_headers)}
proxy = {'https':random.choice(proxy_list)}
def return_html(url):
    r = requests.get(url,headers = header)
    r.encoding='utf-8'
    html = etree.HTML(r.text)
    return html
if __name__ == "__main__":    
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

