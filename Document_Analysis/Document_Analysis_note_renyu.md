参考Tan Yap Peng老师推荐资料简单记一下

## 记录一些资料
### docquery  
2022年一个开源的本地可直接运行的分析文档回答问题模型。  
基于LayoutLM  

### github document-ai主题  
就是随便搜的，不少模型，高星的有：  
**微软的unilm**（一个集大成的git资源库，收集了近年来代表性的文本AI模型，包括LayoutLM、Dit，值得学习)  
ECCV 2022的donut（经典的无OCR端到端Transformer文档理解模型）  
deepdoctection（一个封装好的框架方便做各种文本相关的任务，集成了一些各种任务的经典库）  
awesome-document-understanding（整合了一些经典文档理解任务相关资源的git库）  

### SlideVQA  
AAAI 2023发的一个文档视觉问答数据集，包含了2619个PPT，然后每个PPT20张，一共有14484个QA，890945个边界框。非常适合做PPT分析的任务。  

### document-layout-analysis  
在paperwithcode上搜document layout analysis主题，找到了33篇带code的论文，4个benchmark，9个数据集，似乎还可以。  

### 《A survey of historical document image datasets》  
2022年一篇关于历史文档（就是一些旧的手写手稿和早期印刷品）图片的数据集的综述。做的比较详尽，就是不知道历史文档数据集是否对于简历分析等现代应用有意义。  

### LayoutLM  
2020年的经典文章，微软亚洲研究院搞的第一次在文档理解任务中引入文档布局、样式信息，取得了更好的效果。  

### Review Dit  
ACM MM 2022 经典模型Dit的解析文档  

### 《Document Analysis And Recognition: A survey》  
2021年一篇文档分析的综述  

### LangChain  
LLM应用框架，可以基于LLM的模型做一些应用，例如文档QA、聊天机器人、文本摘要
这个可以研究下，适合我做应用开发  

### resistant.ai  
检测欺诈文档的一个公司，主要是做金融领域一些贷款单、发票、票证的真实性检查
但是应该没有什么开源的东西，只是了解下有假文档检测的这个应用方向  

## 记录一些问题
LLM是decoder-only模型？  
我们是否能训练修改LLM，要多大算力？