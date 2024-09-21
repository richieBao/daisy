import dash
import dash_bootstrap_components as dbc
from dash import Dash, html, dcc,Input, Output,callback,ctx, dash_table, State
from algorithms.heuristicAlgorithm import SMA
from algorithms.heuristicAlgorithm.utils.space import FloatVar
import numpy as np
import sys
# from typing import List, Union, Tuple, Dict
# from ..algorithms.heuristicAlgorithm.utils.problem import Problem
import logging 
import re
import pandas as pd
import plotly.express as px
import plotly.graph_objs as go

import h5py
import pickle
import dash_daq as daq
from dash.exceptions import PreventUpdate

# logger = logging.getLogger(__name__)

# print(dash.get_relative_path("assets/imgs/heuristicAlgorithm/SMA_01.png"))

# class DashLogger(logging.StreamHandler):
#     def __init__(self, stream=None):
#         super().__init__(stream=stream)
#         self.logs = list()

#     def emit(self, record):
#         try:
#             msg = self.format(record)
#             self.logs.append(msg)
#             self.logs = self.logs[-1000:]
#             self.flush()
#         except Exception:
#             self.handleError(record)


# dash_logger = DashLogger(stream=sys.stdout)
# logger.addHandler(dash_logger)

PICKLE_FN='processData/pData_SMA.pkl'
ORIGINAL_POINTS=np.array([[150,5],[25,20],[59,30],[60,10],[70,100]])
popsize=ORIGINAL_POINTS.shape[0]
EPSILON = 10E-10
seed=None
generator = np.random.default_rng(seed)

def  fig_starting_solutions():
    # create image and plotly express object
    fig = px.imshow(
        np.zeros(shape=(90, 160, 4))
    )
    fig.add_scatter(
        x=ORIGINAL_POINTS[:,0],
        y=ORIGINAL_POINTS[:,1],
        mode='markers',
        marker_color='gray',
        marker_size=10,
        name="starting solution"
    )
    fig.add_scatter(
        x=[50],
        y=[50],
        mode='markers',
        marker_color='red',
        marker_size=20,
        marker_symbol="cross",
        name="target"
    )
    
    
    # update layout
    fig.update_layout(
        font_color="blue",
        template='plotly_dark',
        plot_bgcolor='rgba(0, 0, 0, 0)',
        paper_bgcolor='rgba(0, 0, 0, 0)',
        width=700,
        height=500,
        margin={
            'l': 0,
            'r': 0,
            't': 20,
            'b': 0,
        }
    )
    
    
    
    # hide color bar
    fig.update_coloraxes(showscale=False)        
        
    return fig


BLANK_LINE=html.Div(style={'height':'10px'})

fig_staringSolution=fig_starting_solutions()


class DashLogger(logging.StreamHandler):
    def __init__(self, stream=None):
        super().__init__(stream=stream)
        self.logs = list()

    def emit(self, record):
        try:
            msg = self.format(record)
            self.logs.append(msg)
            self.logs = self.logs[-1000:]
            self.flush()
        except Exception:
            self.handleError(record)


def layout_SMA():
    layout=html.Div([
        dbc.Col([
            dbc.Card(
                dbc.CardBody([
                    html.H5("1. SMA——黏菌算法", className="card-title"),
                    html.Hr(),
                    SMA_abstract,
                    
                    ]),
                    className="mt-2"),
            BLANK_LINE,            
            dbc.Card([
                dbc.CardBody([
                        html.H5("2. 参数配置和模拟", className="argsConfig"),
                        # BLANK_LINE,
                        dcc.Markdown(r'''
                                     SMA 算法实现迁移和更新于参考文献\[2]和\[3]的[代码仓库](https://github.com/thieu1995/metaheuristics)。
                                     为了解释算法的机制，将`solutions`（解）的维数定为易于观察的二维，表述为平面坐标系中的点。同时目标函数为简单的距离计算，代码如下：
                                     
                                     ```python
                                     def objective_function(solution):
                                         food = np.array([50,50]) 
                                         result = np.linalg.norm(solution - food)    
                                         
                                         return result                                     
                                     ```
                                     
                                     其中`food = np.array([50,50])`为搜寻的对象（食物），对应为点`[50,50]`。算法模拟提供了可调参数有迭代次数、最大问题或最小问题（试验问题为最小问题），
                                     及概率阈值`p_t`，和初始`solutions`（点）。`输入多个点为起始条件`的最小点数为5。执行`Run（运行）`完成计算，并将相关后续逐步解析所需的数据存入到本地磁盘空间。
                                     
                                     
                                     '''),
                        html.Hr(),
                        args_configSimulation,
                        ]),
                    ]),
            BLANK_LINE,
            dbc.Card([
                dbc.CardBody([
                        html.H5("3. 逐步解析", className="card-title"),
                        dcc.Markdown(r'''     
                                         
                                        SMA 基本算法的伪代码如下：
                                         
                                        1. Algorithm 1 Pseudo-code of SMA                                 
                                        2. Initialize the parameters popsize, Max_iteraition;
                                        3. Initialize the positions of slime mould $X_i(i=1,2, \ldots, n)$;
                                        4. While ( $t \leq$ Max_iteraition)
                                            1. Calculate the fitness of all slime mould;
                                            2. update bestFitness, $X_b$
                                            3. Calculate the $W$ by Eq. (2.5);
                                            4. For each search portion
                                                1. update $p, v b, v c$;
                                                2. update positions by Eq. (2.7);
                                            5. End For
                                            6. $t=t+1$;
                                        5. End While
                                        6. Return bestFitness, $X_b$; 
                                        
                                        试验中，参数`popsize`为`starting solutions`的维度（即种群的大小，多个解组成的集群称为种群`population`），即输入多个点的数量。`Max_iteraition`最大迭代次数可以根据全局最优曲线变化和`fitness`适应度结果来调整。
                                        一个迭代周期中，
                                        
                                        1. 首先逐个计算每个点（agent）到目标点的距离（即由目标函数`objective_function()`计算适应度）；
                                        2. 更新最佳适应度（bestFitness）；
                                        3. 为每一个点（即一个`solution`，解）的 x 和 y 值（即`solution`解中的每个值，1维到多维的向量），计算权重（循环点计算）（公式2.5）；
                                        4. 更新参数 $p$, $\overrightarrow{v b}$ ,$\overrightarrow{v c}$（公式2.4，2.3）；
                                        5. 由$p$为条件，以$\overrightarrow{v b} , \overrightarrow{v c}$为倍数，计算点更新的位置，即更新`solution`（循环点更新）（公式2.1或2.7）；
                                        6. 开始下一个迭代。
                                     
                                     ''',mathjax=True),
                        html.Hr(),
                        
                        stepwiseAnalysis,
                        ]),
                    ]),
            BLANK_LINE,
            dbc.Card([
                dbc.CardBody([
                        html.H6("参考文献：", className="card-title"),
                        # html.Hr(),
                        # html.P("[1] Li, S., Chen, H., Wang, M., Heidari, A. A., & Mirjalili, S. (2020). Slime mould algorithm: A new method for stochastic optimization. Future Generation Computer Systems, 111, 300–323. doi:10.1016/j.future.2020.03.055",
                        #        style={'color': 'gray', 'font-size': '14px'}),
                        dcc.Markdown(r'''
                                     \[1] Li, S., Chen, H., Wang, M., Heidari, A. A., & Mirjalili, S. (2020). Slime mould algorithm: A new method for stochastic optimization. Future Generation Computer Systems, 111, 300–323. doi:10.1016/j.future.2020.03.055.
                                     
                                     \[2] Nguyen, T., Tran, N., Nguyen, B. M., & Nguyen, G. (2018, November). A Resource Usage Prediction System Using Functional-Link and Genetic Algorithm Neural Network for Multivariate Cloud Metrics. In 2018 IEEE 11th Conference on Service-Oriented Computing and Applications (SOCA) (pp. 49-56). IEEE.
                                     
                                     \[3] Nguyen, T., Nguyen, B. M., & Nguyen, G. (2019, April). Building Resource Auto-scaler with Functional-Link Neural Network and Adaptive Bacterial Foraging Optimization. In International Conference on Theory and Applications of Models of Computation (pp. 501-517). Springer, Cham.
                                     
                                     ''',style={'color': 'gray', 'font-size': '14px'}),

                        
                        ]),
                    ]),            
            
            
            
                ])        
                     
        ])
                
    return layout    


SMA_abstract=\
    html.Div([
        dbc.Row([
            
            dbc.Col([
                html.Div([
                    dcc.Markdown(r'''
                                 SMA（slime mould algorithm，黏菌算法）是根据自然界中黏菌的震动模式，提出的一种随机优化算法（stochastic optimizer）。利用自适应权重（adaptive weights）模拟
                                 基于生物振荡器黏菌传播产生的正负反馈过程，用其优秀的探索能力（exploratory ability）和开发倾向（exploitation propensity），
                                 形成连接到食物的最优路径（optimal path），如图 1-1。                                      
                                 '''), 

                ],className="d-grid gap-2",)
                
            ],md=6),
            
            dbc.Col([
                html.Div(children=[
                    html.Img(src=r'/assets/imgs/heuristicAlgorithm/SMA_01.png',alt="SMA",style={'width':'40%'}),
                    html.P("图 1-1 适应度评估（Assessment of fitness）（图片引子参考文献[1]）",style={"color":"gray","font_size":"14px"}),
                    dcc.Markdown(r'''
                                 
                                 
                                 
                                 '''),
                  
                    ],
                    style={"display": "grid"}
                    ),                
            ],md=6)
                        
            ]),  
        
        ],
        # style={'border-bottom':'solid black 1px','display':'grid','grid-template-columns':'600px auto','height':'580px',
        #        'grid-auto-flow':'row'}
    )       
    

args_configSimulation =\
    html.Div([
        dbc.Row([
            
            dbc.Col([
                html.Div([
                    # dbc.Button("应用分析边界", id="save_boundary",n_clicks=0), # ,title="仅使用第1个边界"
                    # html.P(children=["（仅使用绘制的第1个边界）"],className='note',id='save_boundary_info'),   
                    # dbc.Row([
                    #         dbc.Col([dbc.Label("Pop_size（种群数量）:")]),
                    #         dbc.Col([dbc.Input(id="popsize", type="number", value=50)]),
                    #     ]),                       
                    dbc.Row([
                            dbc.Col([dbc.Label("Epoch（迭代次数）:")]),
                            dbc.Col([dbc.Input(id="epoch", type="number", value=20)]),
                        ]),
                    dbc.Row([
                            dbc.Col([dbc.Label("Minmax（最大问题或最小问题）:")]),
                            dbc.Col([dcc.Dropdown(
                               id="minmax",
                               options=['min','max'  ],
                               value="min",
                           )]),
                        ]),                 
                    dbc.Row([
                            dbc.Col([dbc.Label("p_t（probability threshold,概率阈值） [0,1] :")]),
                            dbc.Col([dbc.Input(id="p_t", type="number", value=0.03,min=0,max=10,step=0.01)]),
                        ]),                      
                    
                    
                    dbc.Button("Run（运行）", id="simulating",color="danger", className="me-1",n_clicks=0)
                ],className="d-grid gap-2",)
                
            ],md=4),
            
            dbc.Col([
                html.Div(children=[                    
                    # dcc.Interval(
                    #     id='interval-component',
                    #     interval=1*1000, # in milliseconds
                    #     n_intervals=0
                    # ),
                    # m,
                    # map_controller,      
                    # html.Div(id="graph_starting_solutions"),
                    dbc.Label("输入多个点为起始条件（starting solutions）:"),
                    dcc.RadioItems(
                        id="addDropPts",
                        options=["AddPoints","DropPoints"],
                        value="DropPoints",
                        inline=True,
                        ),
                    dcc.Graph(
                        id='graph_starting_solutions',
                        figure=fig_staringSolution,
                        config={
                            'scrollZoom': True,
                            'displayModeBar': False,
                        }
                    ),
                    
                    html.Div(id='simulation_result', style={'fontSize': 16}),
                    
                    
                    ],
                    style={"display": "grid"}
                    ),                
            ],md=8)
                        
            ]),  
        
        ],
        # style={'border-bottom':'solid black 1px','display':'grid','grid-template-columns':'600px auto','height':'580px',
        #        'grid-auto-flow':'row'}
    )   

stepwiseAnalysis =\
    html.Div([
        dbc.Row([           

            
            dbc.Col([
                dcc.Markdown(r''' 
                             观察每一个迭代由目标函数计算每个点适应度中最佳适应度的变化曲线，并观察点的更新位置，即种群中每个`agent`更新后的`solution`（试验中仅二维，根据分析问题可以为任意维度）。
                             
                             
                             ''',mathjax=True),
                
                html.Div([       
                    dbc.Button("Global Best（全局最优）", id="SMAAnalysis",color="danger", className="me-1",n_clicks=0),
                    BLANK_LINE,
                    html.Div(id="graph_currentGlobalBest"),
                ],className="d-grid gap-2",)
                
            ],md=4),
            
            dbc.Col([
                dcc.Markdown(r'''
                             如果打开`path on`，可以观察每个点更新的路径。从更新路径中可以发现开始迭代时，更新点的跳动幅度较大，但是随着迭代的进行，幅度开始减小并趋向于目标点。这一变化与`绘制曲线 a（arctanh）`，即值$\overrightarrow{v b}$ ,$\overrightarrow{v c}$的变化有关。
                             ''',mathjax=True),
                
                html.Div(children=[
                    # html.Div(id='loggerinfo', style={'fontSize': 16}),
                    # dcc.Graph(id="graph_currentGlobalBest"),
                    # dcc.Slider(1,10,1,value=1,id="slider_epoch")
                    html.Label("位置（solution）更新："),
                    dcc.Slider(0,10,1,id="slider_epoch",value=0,marks=None,tooltip={"placement": "bottom", "always_visible": True}),
                    html.P("path off | path on", style={"textAlign": "center"}),
                    daq.BooleanSwitch(id="pathSwitch", on=False),
                    BLANK_LINE,
                    html.Div(id="graph_solutionPts"),
                    
                    ],
                    style={"display": "grid"}
                    ),                
            ],md=8)
                        
            ]),
        
        html.Hr(),
        
        dbc.Row([
            
            # dbc.Col([
            html.Div([       
                dcc.Markdown(r'''
                             以迭代`epoch`为一轴，对于每一个点 x 和 y 向的更新权重为另外两个轴的值（每个`solution`中各维度上的权重），观察每个点随迭代增加权重的变化。
                             '''),
                
                dbc.Button("Weights-Epoch（权重更新）", id="weights_update",color="danger", className="me-1",n_clicks=0),
                BLANK_LINE,
                html.Div(id="graph_weights"),
                BLANK_LINE,
                
                dcc.Markdown(r'''
                             每次迭代，根据各点的适应度，由解决的问题是最小还是最大问题排序种群中点的顺序。表中给出了各点（种群中的每个代理`agent`）适应度的排序。
                             
                             '''),
                
                html.Label("最优排序索引："),
                html.Div(id="graph_bestSorting"),
                
                
            ],className="d-grid gap-2",),
            # html.Div([   
            #     html.Div(id="graph_bestSorting"),
            #     ],className="d-grid gap-2",),
            
                
            # ],md=4),
            
            # dbc.Col([
                # html.Div(children=[

                    
                    # ],
                    # style={"display": "grid"}
                    # ),                
            # ],md=6)
                        
            ]),     
        
        
        dbc.Row([
            
            dbc.Col([
                html.Div([       
                    BLANK_LINE,
                    dbc.Row([
                            html.Div([dcc.Markdown(
                                    '''
                                    权重计算（公式 2.5,）:        
                                    
                                    ''',mathjax=True)]),
                            BLANK_LINE,           
                            dcc.Markdown(
                                '''
                                $$
                                \\begin{gathered}
                                \\overrightarrow{W(\\text { SmellIndex }(i))}=\\left\\{\\begin{array}{l}
                                1+r \\cdot \\log \\left(\\frac{b F-S(i)}{b F-w F}+1\\right), \\text { condition } \\\\
                                1-r \\cdot \\log \\left(\\frac{b F-S(i)}{b F-w F}+1\\right), \\quad \\text { others }
                                \\end{array}\\right. \\\\
                                \\text { SmellIndex }=\\operatorname{sort}(S)
                                \\end{gathered}
                                    
                                
                                $$
                                
                                权重公式中，$S(i)$为种群的一个解（`solution`）；$bF$为当前迭代中最佳适应度；`wF`为当前迭代中最差适应度。$\\frac{b F-S(i)}{b F-w F}$方法为 Min-Max scaling（最小-最大 标准化）方法，
                                使解的值域缩放至`[0,1]`区间。`log`对数函数则将标准化后的值间的距离拉开，且趋向于最小值一端（左）的值间距离向最大值一端（右）开始拉大，即避免具有较好适应度的多个解的权重被拉开。
                                其中`condition`的条件为，按适应度（fitness）排序种群后（$sort(S)$），前后切分为两份，具有较好适应度部分的解权重值加1，否则用1去减，即具有较好适应度的解具有更高的权重值。
                                
                                ''',mathjax=True
                                ),
                                            
                                    
                            BLANK_LINE,
                            dbc.Col([dbc.Label("迭代索引:")]),
                            dbc.Col([dbc.Input(id="epoch4weightLog", type="number", value=10)]),
                            BLANK_LINE,
                            dbc.Button("绘制曲线 weights （log10）", id="bt_weightsLog",color="danger", className="me-1",n_clicks=0),
                            BLANK_LINE,
                            html.Div(id="graph_weightsLog"),
                            BLANK_LINE,    
                            
                            dbc.Button("绘制曲线 p（tanh）", id="bt_ptanh",color="danger", className="me-1",n_clicks=0),
                            BLANK_LINE,
                            dcc.Markdown(
                                r'''
                                $$p=\tanh |S(i)-D F|$$ （公式 2.2）
                                
                                式中，$S(i)$为一个解的适应度（目标函数返回值），$DF$为经过所有迭代后的最佳适应度（全局适应度）。$p$值表征了当前解到全局适应度的距离，用于公式2.1或2.7的条件。
                                `tanh`函数将距离值缩放至`[0,1]`区间，并拉开各个解之间的距离。当前解越接近于全局最优解，$p$值则越趋近于0，那么生成的随机数$r$，满足$r \geq  p$的可能性越大，更多的是基于自身解的一个更新（$\overrightarrow{v c} \cdot \overrightarrow{X(t)}$）。 
                                而当，当前解距离全局最优解较远，$p$值趋近于1，那么$r < p$的可能性很大，则更多的解更新使用公式$\overrightarrow{X_b(t)}+\overrightarrow{v b} \cdot\left(\vec{W} \cdot \overrightarrow{X_A(t)}-\overrightarrow{X_B(t)}\right)$, 是放弃当前解，而基于当前迭代最优解来更新。
                                
                                
                                
                                ''',mathjax=True
                                ),
                            
                            BLANK_LINE,
                            html.Div(id="graph_ptanh"),
                            
                            
                        ]),
                
                    
                    # dbc.Button("Weights-Epoch（权重更新）", id="weights_update",color="danger", className="me-1",n_clicks=0),
                    # BLANK_LINE,
                    
                    
                    # html.Label("最优排序索引："),
                    # html.Div(id="graph_bestSorting"),
                    
                    
                ],className="d-grid gap-2",),
                # html.Div([   
            #     html.Div(id="graph_bestSorting"),
            #     ],className="d-grid gap-2",),
            
                
            ],md=6),
            
            dbc.Col([
                html.Div(children=[
                    dcc.Markdown(r'''$$a,b \mapsto \overrightarrow{v b}, \overrightarrow{v c}$$：''', mathjax=True),
                    dcc.Markdown(r'''
                                 $$a=\operatorname{arctanh}\left(-\left(\frac{t}{\max _{-} t}\right)+1\right)$$ （公式 2.4）
                                 
                                 $$\overrightarrow{v b}=[-a, a]$$ （公式 2.3）
                                 
                                 $$b=1-\left(\frac{t}{\max _{-} t}\right)$$
                                 
                                 $$\overrightarrow{v c}=[-b, b]$$
                                 
                                 每次迭代时，当前解更新的步幅究竟为多大，对应变量$\overrightarrow{v c}$和$\overrightarrow{v b}$，
                                 其与迭代次数相关，如`绘制曲线a (arctanh)`，当迭代次数增加，变化步幅则逐步减少。$b$为一条斜线，值域`[0,1]`；而$a$在开始迭代时具有较大的值，
                                 即$\overrightarrow{v b}$的区间较大，那么更新的步幅会出现较大的可能性，该种情况主要针对开始迭代期间，当前解不理想的情况。
                                 
                                 
                                 ''',mathjax=True),                  
                                        
                    BLANK_LINE,
                    dbc.Button("绘制曲线 a （arctanh）", id="bt_arctanh",color="danger", className="me-1",n_clicks=0),
                    BLANK_LINE,
                    html.Div(id="graph_arctanh"),
                    BLANK_LINE,
                    dcc.Markdown(r'''
                                 
                                $$
                                \overrightarrow{X(t+1)}=\left\{\begin{array}{c}
                                \overrightarrow{X_b(t)}+\overrightarrow{v b} \cdot\left(\vec{W} \cdot \overrightarrow{X_A(t)}-\overrightarrow{X_B(t)}\right), r<p \\
                                \overrightarrow{v c} \cdot \overrightarrow{X(t)}, r \geq p
                                \end{array}\right.
                                $$
                                 
                                （公式 2.1）
                                
                                
                                $$
                                \overrightarrow{X^*}=\left\{\begin{array}{l}
                                \text { rand } \cdot(U B-L B)+L B, \text { rand }<z \\
                                \overrightarrow{X_b(t)}+\overrightarrow{v b} \cdot\left(W \cdot \overrightarrow{X_A(t)}-\overrightarrow{X_B(t)}\right), r<p \\
                                \overrightarrow{v c} \cdot \overrightarrow{X(t)}, r \geq p
                                \end{array}\right.
                                $$
                                
                                （公式 2.7）
                                
                                由条件$p$，选择更新的方式，当前解趋向于全局最优解时，更多依据自身解更新；否则，更多依据当前最优解，和从种群中随机选择的两个解之间的距离进行更新(包含权重的影响)。
                                $\overrightarrow{v c}$是`[0,1]`区间的一个随机数，且随着迭代次数的增加，区间$[-a,a]$的宽度逐步缩小。
                                
                                
                                ''',mathjax=True),    
                    
                    
                    
                    
                    ],
                    style={"display": "grid"}
                    ),                
            ],md=6),
            
            # dbc.Col([
            #     html.Div(children=[
            #         # dbc.Label("（公式 2.4）:"),
            #         # BLANK_LINE,
                    
            #         ],
            #         style={"display": "grid"}
            #         ),                
            # ],md=3),            
            
            
            
                        
            ]),  
        
        
        
        ],
        # style={'border-bottom':'solid black 1px','display':'grid','grid-template-columns':'600px auto','height':'580px',
        #        'grid-auto-flow':'row'}
    )   





#------------------------------------------------------------------------------



def objective_function(solution):
    food = np.array([50,50]) 
    result = np.linalg.norm(solution - food)    
    
    return result


@callback(
    Output("simulation_result", "children"),
    # Input("popsize","value"),
    Input("epoch","value"),
    Input("minmax","value"),
    Input("p_t","value"),
    Input("simulating","n_clicks"),
    # Input('interval-component', 'n_intervals'),
    )
# def SMA_procedure(problem_dict):
def SMA_procedure(epoch,minmax,p_t,simulating): # ,interval    popsize,
    global problem_dict
    problem_dict = {
        "bounds": FloatVar(lb=(0.,) * 2, ub=(200.,) * 2, name="delta"),
        "minmax": minmax,
        "obj_func": objective_function,
        "log_to":"file",
        "log_file":'loggerfile/logger_SMA.log',
        "return_index":True,
        } 
        
    if simulating>0:
        # print(popsize,epoch,minmax,p_t)
        
        # h5py_fn='processData/pData_SMA.hdf5'
        
        
        
        
        model = SMA.DevSMA(epoch=epoch, pop_size=popsize, p_t = p_t,starting_solutions=ORIGINAL_POINTS,pickle_fn=PICKLE_FN)     # h5py_fn=h5py_fn
        # print(model.pickle_fn)
        g_best = model.solve(problem_dict)
        
        # model=SMA_solve.DevSMA(epoch=epoch, pop_size=popsize, p_t = p_t,starting_solutions=points)  
        
        
        result=f"Solution: {g_best.solution}, Fitness: {g_best.target.fitness}"
        # print(model.logger.name)
        # print(g_best)
        
        # logger = logging.getLogger(model.logger.name)
        # print(logger)
            
        
        return result
        # return [html.Div(log) for log in dash_logger.logs]
        
        # loggerinfo= model.logger.info(f">>>Problem: {self.problem.name}, Epoch: {epoch}, Current best: {self.history.list_current_best[-1].target.fitness}, "
        #                  f"Global best: {self.history.list_global_best[-1].target.fitness}, Runtime: {runtime:.5f} seconds")
        # print(loggerinfo)
    
    # num_points = 10
    # points = np.random.rand(num_points, 2)*10
    # print(points)
    
    # plot_params={}
    # model = SMA.DevSMA(epoch=2, pop_size=6, p_t = 0.03,starting_solutions=points)       
    # g_best = model.solve(problem_dict)
    # print(f"Solution: {g_best.solution}, Fitness: {g_best.target.fitness}")

@callback(
    # Output("loggerinfo", "children"),
    Output("graph_currentGlobalBest","children"),
    Input("SMAAnalysis","n_clicks"),
    )
def SMA_analysis_globalBest(SMAAnalysis):
    if  SMAAnalysis>0:
        try:
            log_file=problem_dict["log_file"]
        except:
            log_file='loggerfile/logger_SMA.log'
        parsed_entries = []
        # print(log_file)
        with open(log_file,'r') as f:
            for line in f:
                match=re.search(r'>>>(.*)',line)
                # print(line)
                # print(match)
                if match:
                    key_value_str = match.group(1).strip()
                    # print(key_value_str)
                    pairs = [kv.strip() for kv in key_value_str.split(',')]
                    # print(pairs)                    
                    key_value_dict = {k.strip(): v.strip() for k, v in (kv.split(':') for kv in pairs)}
                    parsed_entries.append(key_value_dict)
        
        # print(parsed_entries)
        log_df=pd.DataFrame(parsed_entries)
        log_df=log_df.astype({"Epoch":int,"Current_best":float,"Global_best":float})
        # print(log_df,log_df.dtypes)
        fig = px.line(
            log_df,
            x="Epoch",
            y="Global_best",#["Current best","Global best"],
            # title="Epoch - Global best",
        )
        # # fig.update_traces(mode="markers+lines", hovertemplate=None)
        # fig.update_layout(
        #         margin=dict(t=50, l=25, r=25, b=25), yaxis_title="best", xaxis_title="Epoch"
        #     )
        # Update layout
        fig.update_layout(
            title="Epoch - Global best",
            xaxis_title='Epoch',
            yaxis_title='Global best',
            hovermode="x unified"
        ) 
        

        
        
        
        
        # df = px.data.gapminder().query("country=='Canada'")
        # fig = px.line(df, x="year", y="lifeExp", title='Life expectancy in Canada')   
        
        
        return dcc.Graph(figure=fig)
            
@callback(
    Output("slider_epoch","max"),
    Input("epoch","value"),
    )            
def upate_epochnum_max(epoch):
    return epoch
    
@callback(
    Output("graph_solutionPts","children"),
    Input("slider_epoch","value"),
    Input("pathSwitch","on"),
    Input("SMAAnalysis","n_clicks"),
    )    
def graph_solutionPts_update(epoch_slider,on,SMAAnalysis):        
            
        # print("---",epoch)
        if SMAAnalysis>0:
            
            with open(PICKLE_FN,'rb') as f:
                processData_dict=pickle.load(f)
                
            originalPts_df=pd.DataFrame(ORIGINAL_POINTS,columns=["X","Y"])
            originalPts_resetIdx_df=originalPts_df.reset_index().rename({'index':'agent'}, axis = 'columns')
            originalPts_fig=px.scatter(originalPts_resetIdx_df, x="X",y="Y",symbol="agent")
            
            
            order_indices=processData_dict["order_indices"]
            # print(len(order_indices))
            
            ordered_agent={}
            for agent_idx in range(0,popsize):
                agent_idx_copy=agent_idx
                ordered_idx=[]
                for e in range(1,len(order_indices)+1):
                    
                    updated_idx=order_indices[e].index(agent_idx_copy)
                    # print("...",updated_idx)
                    ordered_idx.append(updated_idx)
                    agent_idx_copy=updated_idx
                 
                # print(agent_idx,ordered_idx)
                ordered_agent[agent_idx]=ordered_idx
                
            # print(ordered_agent)    
            global ordered_agent_df
            ordered_agent_df=pd.DataFrame(ordered_agent)
            # print(ordered_agent_df)
            
            
            
            if epoch_slider==0:
                # fig=dcc.Graph(figure=originalPts_fig)
                originalPts_fig.update_traces(marker_size=10)
                return dcc.Graph(figure=originalPts_fig)
            else:

                
                
                
                
                
                pop=processData_dict['pop']
                # print("-"*50)
                # print(len(pop))
                # print(pop[1][0].solution)
                # print([p.solution for p in pop[100]])    
                current_solution=[p.solution for p in pop[epoch_slider]]   
                solution_df=pd.DataFrame(current_solution,columns=['X','Y'])
                solution_df=solution_df.reset_index().rename({'index':'agent'}, axis = 'columns')
                solution_df["agent"]=solution_df["agent"].astype(str)
                
                fig = px.scatter(solution_df, x="X",y="Y",color="agent",symbol="agent")
                
    
                
                fig.add_traces(originalPts_fig.data)
                
                if on:

                    
                    
                    solution4lines=[]
                    for i in range(1,epoch_slider+1):
                        solution_i=[p.solution for p in pop[i]] 
                        solution4line_df=pd.DataFrame(solution_i,columns=['X','Y'])
                        # print(solution4line_df)
                        
                        # print("...",ordered_agent_df[i-1])
                        
                        solution4line_df=solution4line_df.loc[ordered_agent_df.loc[i-1]]
                        solution4line_df=solution4line_df.reset_index(drop=True)
                        
                        
                        solution4lines.append(solution4line_df)
                        
                    # print(solution4lines)
                    
                    
                    solution4lines.insert(0,originalPts_df)
                    merged_df = pd.concat(solution4lines, ignore_index=False)
                    merged_df=merged_df.reset_index().rename({'index':'agent'}, axis = 'columns')
                    # print(merged_df)
                    lines=px.line(merged_df,x='X',y='Y',color='agent')
                    fig.add_traces(lines.data)
                    
                
                fig.update_traces(marker_size=10)
                fig.update_layout(xaxis_range=(0, 200),yaxis_range=(0, 200))
            
                return dcc.Graph(figure=fig)
        # except Exception as e:
            # print(e)
        
@callback(
    Output("graph_weights","children"),
    Output("graph_bestSorting","children"),
    Input("weights_update","n_clicks"),
    # Input("pathSwitch","on"),
    # Input("SMAAnalysis","n_clicks"),
    prevent_initial_call=True,
    )    
def graph_solutionPts_update(weights_update):  
    # print("!"*20)
    # if weights_update==0:
        # pass
    if weights_update>0:
        # print("+"*20)
        
        with open(PICKLE_FN,'rb') as f:
            processData_dict=pickle.load(f)
            
        weights_info=processData_dict["weightEpoch"]
        # print(weights_info)
        
        weights_lst=[]
        for k in range(1,len(weights_info)+1):
            bF=weights_info[k]["bF"]
            wF=weights_info[k]["wF"]
            Si=weights_info[k]["Si"]
            
            ss=bF - wF + EPSILON
            weights=[]
            for idx in range(0,popsize):
                if idx<=int(popsize/2):
                    weight=1+generator.uniform(0,1,2)*np.log10((bF-Si[idx])/ss + 1)
        #             print("+++",weight)
                    weights.append(weight)
                else:
                    weight=1-generator.uniform(0,1,2)*np.log10((bF-Si[idx])/ss + 1)
                    weights.append(weight)                        
                    
            
            # print(weights)
            weights_order=[weights[i] for i in ordered_agent_df.loc[k-1]]
            weights_order_df=pd.DataFrame(weights_order,columns=["X","Y"])
            # print(weights_order)
            weights_order_df['epoch']=k
            # print(weights_order_df)
            
            weights_lst.append(weights_order_df)
        
        # print(weights_dict)
        # print(ordered_agent_df)
        # weights_df=pd.DataFrame(weights_dict)
        # weights_df_T=weights_df.T
        # print(weights_df_T)
        weights_df=pd.concat(weights_lst)
        weights_df=weights_df.reset_index().rename({'index':'agent'}, axis = 'columns')
        weights_df['agent']=weights_df['agent'].astype(str)
        # print(weights_df)
        
        # fig = px.line(
        #     weights_df,
        #     x="X",
        #     y="Y",
        #     color="agent",
        #     markers=True,
        #     title="Weights",
        # )
        # fig.update(layout=dict(title=dict(x=0.5)))
        fig_pts = px.scatter_3d(
            weights_df,
            x="epoch",
            y="X",
            z="Y",
            color="agent",
            # line_dash="dashdot",
            # hover_data=[""],
        )
        fig = px.line_3d(
            weights_df,
            x="epoch",
            y="X",
            z="Y",
            color="agent",
            # line_dash="dashdot",
            # hover_data=[""],
        )
        
        fig.add_traces(fig_pts.data)
        
        
        # print(ordered_agent_df)
        ordered_agent4table_df=ordered_agent_df.rename(columns={col:f"agent_{col}" for col in ordered_agent_df.columns})
        # print(ordered_agent4table_df)
        
        fig_sortingBest=dash_table.DataTable(
            data=ordered_agent4table_df.to_dict("records"),
            columns=[{"name": i, "id": i} for i in ordered_agent4table_df.columns],
            # style_table={'overflowX': 'auto'}
            page_size=10,
            style_data={
                'backgroundColor': 'white',
                'color': 'gray'
            },
            style_header={ 'color': 'blue' },
        )
        
        # print(fig_sortingBest)
        
        
        return [dcc.Graph(figure=fig),fig_sortingBest]
            
            
@callback(
    Output("graph_weightsLog","children"),
    Input("epoch4weightLog","value"),
    Input("bt_weightsLog","n_clicks"),
    prevent_initial_call=True,
    )    
def graph_weightsLog_update(epoch4weightLog,bt_weightsLog):  
    if  bt_weightsLog==0:
        raise PreventUpdate
    if bt_weightsLog>0:        
        with open(PICKLE_FN,'rb') as f:
            processData_dict=pickle.load(f)
            
        weights_info=processData_dict["weightEpoch"]     
        
        chosen_weights_info=weights_info[epoch4weightLog]
        bF=chosen_weights_info["bF"]
        wF=chosen_weights_info["wF"]
        Sis=chosen_weights_info["Si"]
        
        # print(chosen_weights_info)
        x=np.linspace(0, 10, 100)
        base=10
        y=np.log(x+1)/np.log(base)
        
        ss=bF - wF + EPSILON
        chosen_weights=[np.log10((bF-Si)/ss + 1) for Si in Sis]
        
        
        fig = go.Figure()
    
        fig.add_trace(go.Scatter(x=x, y=y, mode='lines', name=f'Log base {base}'))
        fig.add_trace(go.Scatter(x=Sis, y=chosen_weights,name='log(F(S(i)))'))
        fig.add_trace(go.Scatter(x=Sis, y=[0]*len(Sis),name='S(i)'))
        
    
        # Update layout
        fig.update_layout(
            title=f'Logarithmic Curve with Base {base}',
            xaxis_title='X',
            yaxis_title=f'log(F(x)) base {base}',
            hovermode="x unified"
        )        
        
        return dcc.Graph(figure=fig)
    
@callback(
    Output("graph_ptanh","children"),
    Input("epoch4weightLog","value"),
    Input("bt_ptanh","n_clicks"),
    prevent_initial_call=True,
    )    
def graph_ptanh_draw(epoch4weightLog,bt_ptanh):  
    if bt_ptanh>0:        
        with open(PICKLE_FN,'rb') as f:
            processData_dict=pickle.load(f)
            
        weights_info=processData_dict["weightEpoch"]     
        
        chosen_weights_info=weights_info[epoch4weightLog]
        bF=chosen_weights_info["bF"]
        wF=chosen_weights_info["wF"]
        Sis=chosen_weights_info["Si"]    
        
        p =[ np.tanh(np.abs(Si -bF)) for Si in Sis]  
        
        x=np.linspace(-10, 10, 100)
        y=np.tanh(x)
        
        fig = go.Figure()
        
        fig.add_trace(go.Scatter(x=x, y=y, mode='lines', name=f'tanh'))
        fig.add_trace(go.Scatter(x=Sis, y=p,name='p'))
        fig.add_trace(go.Scatter(x=Sis, y=[0]*len(Sis),name='S(i)'))
        
    
        # Update layout
        fig.update_layout(
            title=f'p',
            xaxis_title='X',
            yaxis_title=f'tanh(F(Si))',
            hovermode="x unified"
        )        
        
        return dcc.Graph(figure=fig)        
        
    
    
@callback(
    Output("graph_arctanh","children"),
    Input("bt_arctanh","n_clicks"),
    Input("epoch","value"),
    prevent_initial_call=True,
    )     
def graph_arctanh_draw(bt_arctanh,epoch):
    if bt_arctanh > 0:
        X=list(range(1,epoch+1))
        A=[np.arctanh(-(x / (epoch+1)) + 1) for x in X]
        B=[1 - x / (epoch+1) for x in X]
        
        
        fig = go.Figure()
        fig.add_trace(go.Scatter(x=X, y=A, mode='lines', name=r'a'))
        fig.add_trace(go.Scatter(x=X, y=B, mode='lines', name=r'b'))
        
        # Update layout
        fig.update_layout(
            title=f'a(arctanh); b',
            xaxis_title='X',
            yaxis_title=f'arctanh(F(x))',
            hovermode="x unified"
        ) 
        
        
        return dcc.Graph(figure=fig)
        

@callback(
    Output('graph_starting_solutions', 'figure'),
    State('graph_starting_solutions', 'figure'),
    Input('graph_starting_solutions', 'clickData'),
    Input('addDropPts', 'value'),
)
def get_click(graph_figure, clickData,addDropPts):
    # print("++++",clickData)
    
    if not clickData:
        raise PreventUpdate
    else:
        points = clickData.get('points')[0]
        x = points.get('x')
        y = points.get('y')
        # print("+++",x,y)

        # print(graph_figure['data'])
        # get scatter trace (in this case it's the last trace)
        try:
            scatter_x, scatter_y = [graph_figure['data'][1].get(coords) for coords in ['x', 'y']]       
        
            if addDropPts=="AddPoints":            
                scatter_x.append(x)
                scatter_y.append(y)            
                # print('###',scatter_x,scatter_y)    
                # update figure data (in this case it's the last trace)             
            elif addDropPts=="DropPoints":              
                scatterxy=list(zip(scatter_x, scatter_y))
                if (x,y) in scatterxy:
                    scatterxy.remove((x,y))
                    scatter_x, scatter_y=zip(*scatterxy)
        except:
            scatter_x, scatter_y =[],[]
            
                
        graph_figure['data'][1].update(x=scatter_x)
        graph_figure['data'][1].update(y=scatter_y)          
        

        global ORIGINAL_POINTS
        ORIGINAL_POINTS=np.array(list(zip(scatter_x, scatter_y)))
        # print(ORIGINAL_POINTS)

    return graph_figure   
    



    
    
if __name__=="__main__":
    pass
    # problem_dict = {
    #     "bounds": FloatVar(lb=(0.,) * 2, ub=(10.,) * 2, name="delta"),
    #     "minmax": "min",
    #     "obj_func": objective_function
    #     }
    
    # SMA_procedure(problem_dict)
    
    # -------------------------------------------------------------------------
    # from  algorithms.heuristicAlgorithm.utils import agent
    # a=agent.Agent()
    