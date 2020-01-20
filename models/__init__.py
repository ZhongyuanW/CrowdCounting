# -*- coding: utf-8 -*-
# @Time    : 1/7/20 8:55 PM
# @Author  : zhongyuan
# @Email   : zhongyuandt@gmail.com
# @File    : __init__.py.py
# @Software: PyCharm

from config import MODEL
import models.csrnet.csrnet as net

if MODEL == "csrnet":
    import models.csrnet.csrnet as net
elif MODEL == "dsnet":
    import models.dsnet.dsnet as net
elif MODEL == "mcnn":
    import models.mcnn.mcnn as net
