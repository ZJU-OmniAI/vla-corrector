# VLA-Corrector Video Overview

Generated academic teaser assets for the project page.

| Scene | Visual source | English narration | Chinese narration |
| --- | --- | --- | --- |
| title | Project logo and title card | Meet VLA-Corrector: a lightweight detect-and-correct layer for action-chunked vision-language-action policies. | 这是 VLA-Corrector：一个面向动作块 VLA 策略的轻量级检测与纠错推理框架。 |
| problem | Paper teaser figure: open-loop versus closed-loop execution | Action chunks reduce expensive VLA calls, but they also create an open-loop blind spot. After a small drift, the robot may keep executing stale actions until the horizon ends. | 动作块可以减少昂贵的 VLA 调用，但也带来了开环执行盲区。一次小的偏移之后，机器人可能继续执行已经过时的动作，直到 horizon 结束。 |
| method | Paper method overview figure | VLA-Corrector keeps the backbone policy frozen. A latent-space vision monitor detects visual dynamics mismatch, truncates stale actions, and triggers OGG-guided corrective replanning. | VLA-Corrector 保持 VLA 主干冻结。Latent-space Vision Monitor 检测视觉动态不一致，截断过时动作，并触发 OGG 引导的纠错式重规划。 |
| corrector | Paper truncation phase analysis plus corrector cards | The trainable part is only a small external corrector, about forty million parameters in the paper. It learns local latent dynamics from demonstrations instead of retraining the full VLA. | 可训练部分只是一个外部轻量纠错器。论文中的默认规模约为四千万参数，它从 demonstration 中学习局部 latent dynamics，而不是重新训练完整 VLA。 |
| results | Paper result summary, Pareto figure, and qualitative recovery figure | Across simulation and real-world tasks, the paper reports clear gains: plus fifteen point six five points on MetaWorld with PI zero point five, plus three point eight on LIBERO, and plus seventeen point seven on AgileX PiPER real-world tasks. | 在仿真和真实机器人任务上，论文报告了明显提升：PI 零点五在 MetaWorld 上提升十五点六五个百分点，LIBERO 提升三点八个百分点，AgileX PiPER 真机任务平均提升十七点七个百分点。 |
| demos | Three compressed real-robot demonstration clips | In real-robot disturbance demos, VLA-Corrector is designed to stop trusting stale chunks and recover execution when the object, target, or drawer is moved during the task. | 在真实机器人扰动展示中，当物体、目标碗或抽屉在执行中被移动时，VLA-Corrector 的目标是停止信任过时动作块，并恢复执行。 |
| closing | Project URL and paper status | The code and project page are available now. The paper and arXiv link are coming soon. | 代码和项目主页已经公开。论文和 arXiv 链接即将发布。 |

English duration: 86.7s
Chinese duration: 89.2s

The videos use paper figures and project-owned real-robot clips only. Voice-over is generated with edge-tts.