# ROGII Kaggle Notebook提交约束

核验日期：2026-07-13。只使用Kaggle竞赛官方页面、官方API和官方规则。

| 项目 | 已确认结论 | 官方依据 |
|---|---|---|
| 是否Notebook-only | 是。API返回`isKernelsSubmissionsOnly=true`，Code Requirements要求通过Notebook提交。 | 官方API、Code Requirements |
| 每日提交次数 | 5次；最多选择2个最终提交。 | 官方API、Rules |
| CPU运行时间 | 不超过9小时。 | Code Requirements |
| GPU运行时间 | 不超过9小时。 | Code Requirements |
| CPU/GPU及内存规格 | CPU和GPU Notebook均可使用；具体CPU核心数、GPU型号/数量及RAM上限未确认。 | 竞赛页面未给出精确规格 |
| Internet | 必须关闭。 | Code Requirements |
| 隐藏测试替换 | 是。公开`test/`只是少量训练井样例；Notebook重跑时将替换为约200口真实隐藏测试井。 | Data Description |
| 公开三井性质 | 仅用于编写和验证提交流程的样例，不代表隐藏评分井。 | Data Description |
| submission文件名 | 必须为`submission.csv`，列格式为`id,tvt`。 | Code Requirements、Evaluation |
| submission生成位置 | 必须由提交Notebook运行产生；竞赛专页未明确规定绝对目录，因此绝对路径未确认。 | Code Requirements |

## 模型权重与外部输入

- 官方Code Requirements允许免费、公开可获得的外部数据和预训练模型，Rules要求外部数据/模型对参赛者合理且同等可访问。
- Internet关闭，因此所需权重必须在提交前作为Kaggle托管输入准备，例如创建Kaggle Dataset版本并在提交Notebook中添加为输入，再从该只读输入读取。
- 竞赛专页没有明确说明私有Dataset是否可用于最终重跑，也没有固定权重Dataset名称或绝对挂载路径；这些细节目前标记为**未确认**。
- 权重许可必须满足Competition Rules的外部数据、可访问性和获奖者交付要求。

## 对baseline的直接要求

1. Notebook必须在Internet关闭状态下完整运行。
2. 运行时间预算同时适用于数据读取、特征处理、推理和生成`submission.csv`。
3. 代码必须动态发现隐藏`test/`中的井，不能硬编码公开三井或约200这一数量。
4. 只能把公开三井用于读取、推理、ID对齐和格式验证，不能复制训练标签。

## 官方来源

- [Competition Overview](https://www.kaggle.com/competitions/rogii-wellbore-geology-prediction/overview)
- [Competition Rules](https://www.kaggle.com/competitions/rogii-wellbore-geology-prediction/rules)
- [Competition Data](https://www.kaggle.com/competitions/rogii-wellbore-geology-prediction/data)
- [Code Competition FAQ](https://www.kaggle.com/docs/competitions#notebooks-only-FAQ)
- Kaggle官方API：`competition_list_pages`和`competitions_list`
