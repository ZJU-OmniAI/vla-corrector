# VLA-Corrector 中文宣传稿

## 备选标题

1. 40M 小模型补上机器人开环控制盲区：VLA-Corrector 让 VLA 学会动作纠错
2. 从开环执行到纠错执行：VLA-Corrector 用 40M Corrector 提升机器人鲁棒性
3. 不重训大 VLA，用 40M 小模型修正机器人动作漂移
4. 给 VLA 加一个轻量纠错器：VLA-Corrector 面向具身智能开环盲区
5. 机器人动作跑偏怎么办？VLA-Corrector 用轻量 Corrector 截断 stale action
6. 40M 外部纠错器，让 action-chunked VLA 不再盲目执行长动作片段
7. 面向具身智能的轻量动作纠错：VLA-Corrector 缓解 VLA 开环执行盲区
8. 从固定 action horizon 到自适应纠错：VLA-Corrector 的轻量方案
9. 开环 VLA 的盲区如何补上？一个约 40M 参数的外部 Corrector
10. VLA-Corrector：用轻量潜空间监控与 OGG 引导机器人恢复执行
11. 不改大模型权重，给机器人 VLA 增加动作纠错能力
12. 让 action chunk 更可靠：VLA-Corrector 在执行中发现漂移并触发重规划
13. 机器人长程操作中的动作漂移难题，VLA-Corrector 用 40M 小模型缓解
14. 从开环动作片段到事件触发纠错：VLA-Corrector 面向机器人泛化执行
15. 用潜空间视觉动态监控 VLA：VLA-Corrector 的轻量纠错路径

## 推荐主标题

**40M 小模型补上机器人开环控制盲区：VLA-Corrector 让 VLA 学会动作纠错**

推荐理由：这个标题同时覆盖论文明确支持的约 38--42M 外部 MLP corrector、open-loop blind spot、VLA-Corrector 和动作纠错四个核心点；表述是“补上”和“学会动作纠错”，没有夸大为完整闭环控制系统，也没有使用未确认发表状态或过度宣传词。

## 公众号导语

VLA 让机器人能够把视觉、语言和动作生成统一起来，但在实际执行中，很多策略会一次生成一段 action chunk，再按固定 horizon 开环执行。这样虽然减少了大模型调用次数，却也带来一个盲区：当物体滑动、姿态偏移或环境被扰动时，机器人可能仍在继续执行已经过时的动作。VLA-Corrector 尝试用一个约 40M 参数的轻量外部纠错器，在不重训整个 VLA 的前提下监控潜空间视觉动态、截断 stale action，并用 OGG 引导下一次恢复性重规划，让开环执行具备更及时的纠错能力。

## 正文介绍稿

近年来，Vision-Language-Action（VLA）模型成为具身智能研究的重要方向。它们将视觉观察、语言指令和连续动作生成放在同一框架中，有望支撑更通用的机器人控制。然而，在真正部署到长程、接触丰富的操作任务时，一个工程上很常见的设计会带来新的问题：为了减少大模型推理频率，策略通常一次生成多个未来动作，也就是 action chunk，然后让机器人按固定 action horizon 连续执行。

这种设计可以提升动作连贯性，也能摊薄策略调用成本，但它会形成论文中所说的 open-loop blind spot。机器人执行动作片段时，新观察虽然持续到来，但策略并不会在每一步重新使用这些观察。如果物体发生滑移、碰撞、姿态漂移，或者人手临时移动了目标，机器人可能仍然沿着旧动作队列前进。偏差一旦在盲区内累积，就可能把机器人推到训练中很少见的状态，下一次普通重规划也未必能自然恢复。

VLA-Corrector 的核心思路不是重训整个大 VLA，而是在推理阶段增加一个轻量外部纠错路径。论文中的 corrector 是一个用于短时潜空间视觉动态预测的残差 MLP，实验中约为 38--42M 参数，因此可概括为约 40M 的轻量 MLP corrector。它在 VLA 主干已经获得后训练：先冻结 VLA，用视觉编码器从演示轨迹中提取潜在特征，再学习当前视觉潜变量和动作会引起怎样的短时潜空间残差。

部署时，VLA-Corrector 使用 Latent-space Vision Monitor（LVM）比较“预期的视觉动态”和“真实观察到的视觉动态”。如果不一致性只是偶发波动，系统不会立即干预；只有当动态阈值和持续性检测都表明偏差在累积时，才触发 interrupt event。触发后，系统会丢弃当前 action chunk 中剩余的 stale action，把原本固定的 action horizon 变成事件触发的自适应 horizon。

仅仅截断旧动作还不够，下一次重规划也需要更偏向恢复方向。因此，VLA-Corrector 在中断后的单次策略调用中使用 Online Gradient Guidance（OGG），利用预测与观测之间的潜空间动态差异，引导新的动作生成朝更有利于恢复的方向移动。稳定阶段仍保持普通 action chunk 执行，只有检测到持续漂移时才额外付出纠错计算。

论文在 MetaWorld、LIBERO 以及 AgileX PiPER 6-DoF 真机平台上进行了验证。MetaWorld 上，VLA-Corrector 在 PI0.5、SmolVLA 和 X-VLA 三类 backbone 上均带来平均成功率提升；LIBERO 中，few-shot fine-tuned PI0.5 加上 VLA-Corrector 后，论文报告的平均成功率从 94.00% 提升到 97.80%；真机实验中，方法在 pick-and-place、alignment 和 disturbance recovery 三组任务上提升平均成功率，其中对扰动恢复任务的收益最大。

这些结果说明，开环动作片段的盲区并不一定只能通过更频繁地调用大模型或收集大量恢复数据来缓解。一个外部、轻量、面向潜空间动态一致性的 corrector，也可以在不改动 VLA 主干权重的前提下，为机器人执行增加事件触发的动作纠错能力。当然，论文也明确指出，OGG 会带来额外推理开销，且 VLA-Corrector 不能创造主干策略本身无法表达的恢复行为；它更适合作为冻结 VLA 之上的推理时鲁棒性增强模块。

项目代码已经开源，包含 LeRobot 代码基础上的修改版评测入口、PI0.5 / SmolVLA / X-VLA 相关策略包装、潜空间动态 corrector 训练与推理模块，以及 GitHub Pages 项目主页。数据集、预训练权重、微调 checkpoint 和演示数据不会随仓库发布，需要用户根据 README 自行准备。

## 精简版摘要

VLA-Corrector 面向 action-chunked VLA 的开环执行盲区：它用约 40M 参数的轻量外部 corrector 监控潜空间视觉动态，在检测到持续漂移时截断 stale action，并用 OGG 引导恢复性重规划。不重训大 VLA，也不上传权重和数据，目标是在推理阶段为机器人执行增加更及时的动作纠错能力。

## 项目链接区

```text
代码：https://github.com/ZJU-OmniAI/vla-corrector
主页：https://zju-omniai.github.io/vla-corrector/
论文：Coming soon
arXiv：Coming soon
```
