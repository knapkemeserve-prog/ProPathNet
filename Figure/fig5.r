# 如果没有安装这些包，请先运行: install.packages(c("ggplot2", "dplyr"))
library(ggplot2)
library(dplyr)

# 1. 读取你从 g:Profiler 下载的 CSV 文件 (请将 "your_file.csv" 替换为实际的文件路径)
# 注意：g:Profiler 导出的 CSV 可能不带表头或者包含注释行，如果读取报错，
# 可以尝试加上参数：read.csv("your_file.csv", comment.char = "#")
#data <- read.csv("/home/xuln/olink_disease_predict/code/gProfiler_hsapiens_Primary_Malignancy_Prostate_2026-05-08.csv")
#data <- read.csv("/home/xuln/olink_disease_predict/gprofiltered/gProfiler_hsapiens_2026-05-26_Cancers_top15.csv")
#data <- read.csv("/home/xuln/olink_disease_predict/gprofiltered/Cancers_select_protein_80.csv")
#data <- read.csv("/home/xuln/olink_disease_predict/gprofiltered/Digestive_select_protein_91.csv")
#data <- read.csv("/home/xuln/olink_disease_predict/gprofiltered/Psychiatric_select_protein_53.csv")
#data <- read.csv("/home/xuln/olink_disease_predict/gprofiltered/Infections_select_protein_81.csv")


#data <- read.csv("/home/xuln/olink_disease_predict/code/gProfiler_hsapiens_Osteoporosis_2026-05-08.csv")
#data <- read.csv("/home/xuln/olink_disease_predict/code/gProfiler_hsapiens_Iron_deficiency_anaemia_2026-05-08.csv")
data <- read.csv("/home/xuln/olink_disease_predict/code/gProfiler_hsapiens_Stable_angina_2026-05-08.csv")
#data <- read.csv("/home/xuln/olink_disease_predict/gprofiltered/gProfiler_hsapiens_Hypertension_2026-05-08.csv")


# 2. 数据清洗与预处理 (这一部分与条形图完全一致)
plot_data <- data %>%
  # 仅保留 GO 的三个主要分支
  filter(source %in% c("GO:BP", "GO:CC", "GO:MF")) %>%
  
  # 重命名标签
  mutate(source = factor(source, levels = c("GO:BP", "GO:CC", "GO:MF"), labels = c("BP", "CC", "MF"))) %>%
  
  # 根据 p 值升序排列
  group_by(source) %>%
  arrange(adjusted_p_value) %>%
  
  # 提取 Top 10 最显著的通路
  slice_head(n = 10) %>%
  ungroup() %>%
  
  # 因子化排序，让它在 Y 轴上按照基因数和 source 有序排列
  arrange(source, intersection_size) %>%
  mutate(term_name = factor(term_name, levels = unique(term_name)))

# 3. 使用 ggplot2 绘制分面气泡图
p <- ggplot(plot_data, aes(x = intersection_size, y = term_name)) +
  
  geom_point(aes(size = intersection_size, color = adjusted_p_value)) + 
  
  facet_grid(source ~ ., scales = "free_y", space = "free_y") +
  
  scale_color_gradient(low = "#e41a1c", high = "#377eb8", name = "p.adj", 
                       guide = guide_colorbar(order = 2)) +
  
  scale_size_continuous(range = c(3, 8), name = "Gene Count", 
                        guide = guide_legend(order = 1)) +
  
  # 【修改1】：在 labs 中增加 title 参数设置标题文字
  labs(
    title = "Stable angina", 
    x = "Gene Count (Intersection)", 
    y = NULL
  ) +
  
  scale_y_discrete(labels = scales::label_wrap(width = 50)) +

  # 应用学术主题设置
  theme_bw() +
  theme(
    panel.grid.major = element_blank(), 
    panel.grid.minor = element_blank(),
    
    # 【修改2】：在这里控制标题的居中 (hjust = 0.5) 以及字体大小加粗
    plot.title = element_text(hjust = 0.5, size = 18, face = "bold"),
    
    # 原有的字体大小和颜色设置保持不变
    axis.text.y = element_text(size = 14, color = "black"),
    axis.text.x = element_text(size = 14, color = "black"),
    axis.title.x = element_text(size = 14, face = "bold"),
    
    strip.text = element_text(size = 16, face = "bold", angle = -90), 
    strip.background = element_rect(fill = "grey95", color = "black"),
    
    panel.border = element_rect(color = "black", fill = NA, linewidth = 0.8),
    legend.position = "right"
  )

# 渲染图表
print(p)

# 4. 导出图表（稍微修改了文件名，加上了 _bubble 防止覆盖原图）
ggsave(
  filename = "/home/xuln/olink_disease_predict/gprofiltered/GO_enrichment_bubble_Stable_angina.png", 
  plot = p, 
  width = 10, 
  height = 10, 
  units = "in",
  dpi = 300,
  bg = "white"
)