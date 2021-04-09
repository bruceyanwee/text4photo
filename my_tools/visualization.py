from pyecharts import options as opts
from pyecharts.charts import Bar
from pyecharts.render import make_snapshot

from snapshot_selenium import snapshot

def bar_chart() -> Bar:
    c = (
        Bar()
            .add_xaxis(["衬衫", "毛衣", "领带", "裤子", "风衣", "高跟鞋", "袜子"])
            .add_yaxis("商家A", [114, 55, 27, 101, 125, 27, 105])
            .add_yaxis("商家B", [57, 134, 137, 129, 145, 60, 49])
            .reversal_axis()
            .set_series_opts(label_opts=opts.LabelOpts(position="right"))
            .set_global_opts(title_opts=opts.TitleOpts(title="Bar-测试渲染图片"))
    )
    return c

import pyecharts.options as opts
from pyecharts.charts import Radar

"""
Gallery 使用 pyecharts 1.1.0
参考地址: https://echarts.baidu.com/examples/editor.html?c=radar

目前无法实现的功能:

1、雷达图周围的图例的 textStyle 暂时无法设置背景颜色
"""
v1 = [[0.8300, 0.10000, 0.28000, 0.35000, 0.30000, 0.19000,0.1000,0.2300]]
v2 = [[0.5000, 0.14000, 0.88000, 0.31000, 0.42000, 0.21000,0.1200,0.0800]]
v3 = [[0.69000, 0.13000, 0.8000, 0.42000, 0.48000, 0.23000,0.1300,0.1700]]
(
    Radar(init_opts=opts.InitOpts(width="980px", height="820px", bg_color="#999999"))
        .add_schema(
        schema=[
            opts.RadarIndicatorItem(name="有趣", max_=1),
            opts.RadarIndicatorItem(name="敬畏", max_=1),
            opts.RadarIndicatorItem(name="愉快", max_=1),
            opts.RadarIndicatorItem(name="满足", max_=1),
            opts.RadarIndicatorItem(name="愤怒", max_=1),
            opts.RadarIndicatorItem(name="恶心", max_=1),
            opts.RadarIndicatorItem(name="恐怖", max_=1),
            opts.RadarIndicatorItem(name="悲伤", max_=1)
        ],
        splitarea_opt=opts.SplitAreaOpts(
            is_show=True, areastyle_opts=opts.AreaStyleOpts(opacity=1)
        ),
        textstyle_opts=opts.TextStyleOpts(color="#fff"),
    )
        .add(
        series_name="仅照片情感",
        data=v1,
        linestyle_opts=opts.LineStyleOpts(color="#CD0000"),
    )
        .add(
        series_name="仅配文情感",
        data=v2,
        linestyle_opts=opts.LineStyleOpts(color="#5CACEE"),
    )
        .add(
        series_name="多模态情感",
        data=v3,
        linestyle_opts=opts.LineStyleOpts(color="#9CACEE"),
    )
        .set_series_opts(label_opts=opts.LabelOpts(is_show=False))
        .set_global_opts(
        title_opts=opts.TitleOpts(title="情感识别可视化"), legend_opts=opts.LegendOpts()
    )
        .render("sentiment_radar.html")
)

#  词云效果图
import pyecharts.options as opts
from pyecharts.charts import WordCloud
from pyecharts.globals import SymbolType
"""
Gallery 使用 pyecharts 1.1.0
参考地址: https://gallery.echartsjs.com/editor.html?c=xS1jMxuOVm

目前无法实现的功能:

1、暂无
"""

data_one_shot = [
    ("草坪", "10"),
    ("白云",'8'),
    ("金毛",'9'),
    ("台阶",'6'),
    ("青草",'12'),
]
data_text_prefer = [
    ("古文",'20'),
    ("回忆",'39'),
    ("坚定",'12'),
    ("婚姻",'14'),
    ("安慰",'29'),
    ("寂寞",'10'),
    ("工作",'20'),
    ("怀旧",'37'),
]
data_photo_prefer = [
    ("宠物",'20'),
    ("金毛",'39'),
    ("街道",'12'),
    ("马路",'14'),
    ("樱花",'29'),
    ("树叶",'10'),
    ("银杏",'20'),
    ("室外",'88'),
    ("海滩",'38'),
]
(
    WordCloud()
        .add(series_name="照片内容", data_pair=data_one_shot, word_size_range=[16, 36],shape=SymbolType.DIAMOND)
        .set_global_opts(
        title_opts=opts.TitleOpts(
            title="照片内容", title_textstyle_opts=opts.TextStyleOpts(font_size=23)
        ),
        tooltip_opts=opts.TooltipOpts(is_show=True),
    )
        .render("data_one_shot.html")
)
(
    WordCloud()
        .add(series_name="配文偏好", data_pair=data_text_prefer, word_size_range=[16, 36],shape=SymbolType.DIAMOND)
        .set_global_opts(
        title_opts=opts.TitleOpts(
            title="配文偏好", title_textstyle_opts=opts.TextStyleOpts(font_size=23)
        ),
        tooltip_opts=opts.TooltipOpts(is_show=True),
    )
        .render("data_text_prefer.html")
)
(
    WordCloud()
        .add(series_name="拍照偏好", data_pair=data_photo_prefer, word_size_range=[16, 36],shape=SymbolType.DIAMOND)
        .set_global_opts(
        title_opts=opts.TitleOpts(
            title="拍照偏好", title_textstyle_opts=opts.TextStyleOpts(font_size=23)
        ),
        tooltip_opts=opts.TooltipOpts(is_show=True),
    )
        .render("data_photo_prefer.html")
)

from pyecharts import options as opts
from pyecharts.charts import Pie
from pyecharts.faker import Faker

pie_data = [('其他',30),
            ('一个人',40),
            ('火锅',50),
            ('知足',60),
            ('怀念',20),
            ('旅行',60),
            ('净化',30),
            ('舒服',30),
            ('开心',30),]
pie_data_2 = [('狗',30),
            ('街道',40),
            ('樱花',50),
            ('梅花',60),
            ('沙滩',20),
            ('树木',60),
            ('人像',30),
            ('路灯',30),
            ('星空',20),
            ('其他',20),
            ]
c = (
    Pie()
        .add(
        "",
        pie_data_2,
        radius=["40%", "55%"],
        label_opts=opts.LabelOpts(
            position="outside",
            formatter="{a|{a}}{abg|}\n{hr|}\n {b|{b}: }{c}  {per|{d}%}  ",
            background_color="#eee",
            border_color="#aaa",
            border_width=1,
            border_radius=4,
            rich={
                "a": {"color": "#999", "lineHeight": 22, "align": "center"},
                "abg": {
                    "backgroundColor": "#e3e3e3",
                    "width": "100%",
                    "align": "right",
                    "height": 22,
                    "borderRadius": [4, 4, 0, 0],
                },
                "hr": {
                    "borderColor": "#aaa",
                    "width": "100%",
                    "borderWidth": 0.5,
                    "height": 0,
                },
                "b": {"fontSize": 16, "lineHeight": 33},
                "per": {
                    "color": "#eee",
                    "backgroundColor": "#334455",
                    "padding": [2, 4],
                    "borderRadius": 2,
                },
            },
        ),
    )
        .set_global_opts(title_opts=opts.TitleOpts(title="Pie-富文本示例"))
        .render("pie_rich_label.html")
)

# 拍照偏好
# 配文偏好

# 画分布图
# make_snapshot(snapshot, bar_chart().render(), "bar0.png")
# # 雷达图 情感概率
# make_snapshot(snapshot, "sentiment_radar.html", "../visual_results/sentiment_radar.png")
# # 词云可视化
# make_snapshot(snapshot, "data_one_shot.html", "../visual_results/data_one_shot.png")
# make_snapshot(snapshot, "data_text_prefer.html", "../visual_results/data_text_prefer.png")
# make_snapshot(snapshot, "data_photo_prefer.html", "../visual_results/data_photo_prefer.png")

make_snapshot(snapshot, "pie_rich_label.html", "../visual_results/pie_rich_label.png")


