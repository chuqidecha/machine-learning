
# seaborn学习笔记-统计关系可视化之relplot

```python
seaborn.relplot(x=None, y=None, hue=None, size=None, style=None, data=None, row=None, col=None, col_wrap=None, row_order=None, col_order=None, palette=None, hue_order=None, hue_norm=None, sizes=None, size_order=None, size_norm=None, markers=None, dashes=None, style_order=None, legend='brief', kind='scatter', height=5, aspect=1, facet_kws=None, **kwargs)
```
参数：x,y: data中的列名
   - 输入变量，必须是数值型    
   hue：data中的列名（可选）        
   - 用不同颜色对变量分组，类别或者数值
   size: data中的列名（可选）    
   - 用不同的大小对变量分组，类别或者数值
   style: data中的列名（可选）    
   - 用不同的仰视对变量分组，可以是数值但会被视为类别
   row,col: data中的列名（可选）    
   - 用于决定Facegrid中的每个格网切面
   col_wrap: 整型（可选）    
   - 自定义Facegrid中各个图的行列排序方式    
   row_order,col_orde:字符串列表（可选）    
   - Facegrid中每行的列数    
   platte: 调色板，字符串列表或者字典（可选）   
   - hue变量使用的调色板       
   hue_order: 列表（可选）
   - 自定义调色板中颜色的选择方式，当hue为数值型时，该变量无效    
   hue_norm: 元组或者归一化对象（可选）
   - 归一化hue,当hue为类别时，该变量无效    
   sizes：字典、列表或元组（可选）
   - 当设置size变量时用于控制大小    
   size_order: 列表（可选）
   - 自定义sizes的选择方式，当size为数值型时，该变量无效    
   size_norm： 元组或者归一化对象（可选）
   - 归一化size
   legend: "brief"、"full"或则False（可选）
   - 图例,对于数值型变量，brief仅显示部分    
   kind: string（可选)
   - line或者scatter    
   height: 数值常量
   - 字图的高度    
   aspect: 数值常量
   - 宽高比 height*aspect 为字图宽度   
   facet_kws：字典（可选)
   - 传递给FacetGrid的参数
   kwargs: 键值对
   - 其他参数   
   
返回值: g:FacetGrid
   - 返回FacetGrid    
   
replot函数是用于在FacetGrid上绘制关系图的Figure-level函数。此函数提供了对几个不同axes-level函数的访问，这些axes-level函数通过语义映射显示数据集中多个变量之间的关系。kind参数用于控制使用哪个axes-level函数：
* scatterplot() (kind="scatter",默认情况)
* lineplot() (kind="line")


```python
import warnings
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
sns.set(style="darkgrid")
```

## 散点图


```python
tips = sns.load_dataset("tips")
tips.head()
```




<div>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>total_bill</th>
      <th>tip</th>
      <th>sex</th>
      <th>smoker</th>
      <th>day</th>
      <th>time</th>
      <th>size</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>16.99</td>
      <td>1.01</td>
      <td>Female</td>
      <td>No</td>
      <td>Sun</td>
      <td>Dinner</td>
      <td>2</td>
    </tr>
    <tr>
      <th>1</th>
      <td>10.34</td>
      <td>1.66</td>
      <td>Male</td>
      <td>No</td>
      <td>Sun</td>
      <td>Dinner</td>
      <td>3</td>
    </tr>
    <tr>
      <th>2</th>
      <td>21.01</td>
      <td>3.50</td>
      <td>Male</td>
      <td>No</td>
      <td>Sun</td>
      <td>Dinner</td>
      <td>3</td>
    </tr>
    <tr>
      <th>3</th>
      <td>23.68</td>
      <td>3.31</td>
      <td>Male</td>
      <td>No</td>
      <td>Sun</td>
      <td>Dinner</td>
      <td>2</td>
    </tr>
    <tr>
      <th>4</th>
      <td>24.59</td>
      <td>3.61</td>
      <td>Female</td>
      <td>No</td>
      <td>Sun</td>
      <td>Dinner</td>
      <td>4</td>
    </tr>
  </tbody>
</table>
</div>



1. 散点图是统计可视化的基础。它使用点云描述两个变量的联合分布，其中每个点表示数据集中的一个观察值。


```python
sns.relplot(x="total_bill", y="tip", data=tips)
```

![png](output_6_1.png)


2. 尽管散点图是二维的，但是可以通过给点着色来表示第三维变量。


```python
sns.relplot(x="total_bill", y="tip", hue="smoker", data=tips)
```


![png](output_8_0.png)


3. 为了强调类别之间的差异，并提高可识别性，可以对每个类别使用不同的样式。


```python
sns.relplot(x="total_bill", y="tip", hue="smoker", style="smoker",data=tips)
```


![png](output_10_1.png)


4. 通过独立地改变每个点的颜色和样式来表示四个变量。


```python
sns.relplot(x="total_bill", y="tip", hue="smoker", style="time", data=tips)
```


![png](output_12_1.png)


5. 如果hue是连续数值而不是离散的，则颜色的深浅代表不同的取值


```python
sns.relplot(x="total_bill", y="tip", hue="size", data=tips)
```


![png](output_14_1.png)


6. 也可以自定义颜色


```python
sns.relplot(x="total_bill", y="tip", hue="size", palette="ch:r=-.5,l=.75", data=tips)
```

![png](output_16_1.png)


7. 除了颜色和样式，大小也可以表示维度


```python
sns.relplot(x="total_bill", y="tip", size="size", data=tips)
```

![png](output_18_1.png)


8. 可以自定义大小


```python
sns.relplot(x="total_bill", y="tip", size="size", sizes=(15, 200), data=tips)
```


![png](output_20_1.png)


## 折线图


```python
df = pd.DataFrame(dict(time=np.arange(500),value=np.random.randn(500).cumsum()))
df.head()
```




<div>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>time</th>
      <th>value</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>0</td>
      <td>-1.628097</td>
    </tr>
    <tr>
      <th>1</th>
      <td>1</td>
      <td>-0.481164</td>
    </tr>
    <tr>
      <th>2</th>
      <td>2</td>
      <td>-1.086899</td>
    </tr>
    <tr>
      <th>3</th>
      <td>3</td>
      <td>-0.686817</td>
    </tr>
    <tr>
      <th>4</th>
      <td>4</td>
      <td>-0.976437</td>
    </tr>
  </tbody>
</table>
</div>



1. 在某些情况下需要观察一个连续变量随着另一个变量的变化时，折线图是一个很好的选择。在seaborn中通过设置relplot的kind参数值为line，可以绘制折线图。


```python
g = sns.relplot(x="time", y="value", kind="line", data=df)
g.fig.autofmt_xdate()
```


![png](output_24_0.png)


默认情况下会对x进行排序，通过设置sort参数的值为False来禁止排序。


```python
df = pd.DataFrame(np.random.randn(500, 2).cumsum(axis=0), columns=["x", "y"])
sns.relplot(x="x", y="y", sort=False, kind="line", data=df)
```

![png](output_26_1.png)


2. 聚合和不确定性表示


```python
fmri = sns.load_dataset("fmri")
fmri.head()
```




<div>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>subject</th>
      <th>timepoint</th>
      <th>event</th>
      <th>region</th>
      <th>signal</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>s13</td>
      <td>18</td>
      <td>stim</td>
      <td>parietal</td>
      <td>-0.017552</td>
    </tr>
    <tr>
      <th>1</th>
      <td>s5</td>
      <td>14</td>
      <td>stim</td>
      <td>parietal</td>
      <td>-0.080883</td>
    </tr>
    <tr>
      <th>2</th>
      <td>s12</td>
      <td>18</td>
      <td>stim</td>
      <td>parietal</td>
      <td>-0.081033</td>
    </tr>
    <tr>
      <th>3</th>
      <td>s11</td>
      <td>18</td>
      <td>stim</td>
      <td>parietal</td>
      <td>-0.046134</td>
    </tr>
    <tr>
      <th>4</th>
      <td>s10</td>
      <td>18</td>
      <td>stim</td>
      <td>parietal</td>
      <td>-0.037970</td>
    </tr>
  </tbody>
</table>
</div>



复杂的数据集会有多种方式测量同一个变量x的值。seaborn中默认的方式是在每一个x处通过绘制均值和95%的置信区间来聚合这些测量值。


```python
sns.relplot(x="timepoint", y="signal", kind="line", data=fmri)
```

![png](output_30_1.png)


置信区间是通过bootstrapping方法计算的，对于较大的数据集计算比较耗时，通过ci参数可以禁用置信区间。


```python
sns.relplot(x="timepoint", y="signal", ci=None, kind="line", data=fmri)
```

![png](output_32_1.png)


对于较大的数据集另一种方法是通过绘制标准偏差而不是置信区间来表示每个时间点的分布。


```python
sns.relplot(x="timepoint", y="signal", kind="line", ci="sd", data=fmri)
```

![png](output_34_1.png)


通过设置estimator为None可以禁用聚合功能，但当同一位置有多个观测值时，图形可能看起来很奇怪。


```python
sns.relplot(x="timepoint", y="signal", estimator=None, kind="line", data=fmri)
```

![png](output_36_1.png)


3. 使用语义映射绘制数据集的子集


```python
sns.relplot(x="timepoint", y="signal", hue="event", kind="line", data=fmri)
```


![png](output_38_1.png)


与散点图相同，折线图也可以通过改变颜色、大小和样式来表示更多的变量。


```python
sns.relplot(x="timepoint", y="signal", hue="region", style="event",kind="line", data=fmri)
```




    <seaborn.axisgrid.FacetGrid at 0x7f167c3ae9e8>




![png](output_40_1.png)


使用重复测量数据（即有多次采样的单位）时，可以单独绘制每个采样单位，比如对”subject”单位绘图。


```python
sns.relplot(x="timepoint", y="signal", hue="region",units="subject", estimator=None,kind="line", data=fmri.query("event == 'stim'"));
```


![png](output_42_0.png)


默认的色彩映射和图例的处理还取决于色调语义是分类还是数字


```python
dots = sns.load_dataset("dots").query("align == 'dots'")
dots.head(4)
sns.relplot(x="time", y="firing_rate",hue="coherence", style="choice",kind="line", data=dots)
```




    <seaborn.axisgrid.FacetGrid at 0x7f167c28e940>




![png](output_44_1.png)


通过传递列表或字典为每一行提供特定的颜色值


```python
palette = sns.cubehelix_palette(light=.8, n_colors=6)
sns.relplot(x="time", y="firing_rate",hue="coherence", style="choice",palette=palette,kind="line", data=dots);
```


![png](output_46_0.png)


4. 时间序列绘图


```python
df = pd.DataFrame(dict(time=pd.date_range("2017-1-1", periods=500),value=np.random.randn(500).cumsum()))
g = sns.relplot(x="time", y="value", kind="line", data=df)
g.fig.autofmt_xdate()
```


![png](output_48_0.png)


## 利用facet显示多元关系

1. 通过col参数绘制多列


```python
sns.relplot(x="total_bill", y="tip", hue="smoker",col="time", data=tips)
```

![png](output_51_1.png)


2. 通过row参数设置多行


```python
sns.relplot(x="timepoint", y="signal", hue="subject",col="region", row="event", height=3,kind="line", estimator=None, data=fmri)
```


![png](output_53_1.png)


3. 通过col_wrap参数设置列数,height参数设置高度，aspect设置宽高比，linewidth设置线宽


```python
sns.relplot(x="timepoint", y="signal", hue="event", style="event",col="subject", col_wrap=5,height=3, aspect=.75, linewidth=2.5,kind="line", data=fmri.query("region == 'frontal'"))
```


![png](output_55_1.png)

