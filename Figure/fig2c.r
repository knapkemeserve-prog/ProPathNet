#这个版本的代码缩写了一下疾病的名称，疾病类型过长的该换行的换行，然后增加一点每一行的行高

# ==============================================================================
# 1. 加载包
# ==============================================================================
library(grid)
library(forestploter)
library(dplyr)
library(tidyr)
library(gridExtra)

# ==============================================================================
# 2. 数据准备 - 读取C-index汇总结果和差异比较结果
# ==============================================================================

comparison_path <- "/bigdat2/user/xuln/olink_disease_predict/67traits_cox_analysis/all_diseases_bootstrap_summary_combined.csv"
comparison_data <- read.csv(comparison_path)

# ==============================================================================
# 3. 筛选指定的疾病类型
# ==============================================================================

if (!"disease_category" %in% colnames(comparison_data)) {
  stop("数据中没有disease_category列，请检查数据文件!")
}

target_categories <- c("Cancers", "Digestive", "Infections", "Psychiatric","Haematological_or_immunological")
filtered_data <- comparison_data %>% 
  filter(disease_category %in% target_categories)

# ==============================================================================
# 智能缩写长名称
# ==============================================================================

# 确定疾病名称列名
disease_cols <- grep("disease|Disease|trait|Trait", colnames(filtered_data), value = TRUE, ignore.case = TRUE)
if ("disease_name" %in% colnames(filtered_data)) {
  disease_col <- "disease_name"
} else if ("Disease" %in% colnames(filtered_data)) {
  disease_col <- "Disease"
} else if (length(disease_cols) > 0) {
  disease_col <- disease_cols[1]
} else {
  filtered_data$Disease_Seq <- paste0("Disease_", 1:nrow(filtered_data))
  disease_col <- "Disease_Seq"
}

filtered_data <- filtered_data %>%
  mutate(
    # 缩写超长的疾病名称 (Disease) 
    !!sym(disease_col) := gsub("and subcutaneous tissue", "& Subcutaneous", !!sym(disease_col), ignore.case = TRUE),
    !!sym(disease_col) := gsub("Liver and intrahepatic bile duct", "Liver & Bile Duct", !!sym(disease_col), ignore.case = TRUE),
    !!sym(disease_col) := gsub("Delirium, not induced by alcohol and other psychoactive substances", "Delirium (Non-substance)", !!sym(disease_col), ignore.case = TRUE)
  )

# ==============================================================================
# 4. 构建绘图表格 - 双重排序
# ==============================================================================

# 按疾病类型(A-Z)排序，再在同类型内按疾病名称(A-Z)排序
combined_data <- filtered_data %>%
  arrange(disease_category, .data[[disease_col]])

# 格式化 C-index
combined_data$Cindex_Str <- ifelse(
  !is.na(combined_data$cindex_mymodel_mean),
  paste0(
    sprintf("%.2f", combined_data$cindex_mymodel_mean), " [",
    sprintf("%.2f", combined_data$cindex_mymodel_ci_lower), "-",
    sprintf("%.2f", combined_data$cindex_mymodel_ci_upper), "]"
  ),
  "Data not available"
)

# 创建绘图数据框（删除 Clinic+ ProPathNet Riskscore 列）
plot_data <- data.frame(
  `Disease Type` = combined_data$disease_category,  
  Disease = trimws(combined_data[[disease_col]]),
  `ProPathNet C-index [95% CI]` = combined_data$Cindex_Str, 
  `Base` = paste(rep(" ", 30), collapse = " "),
  `Clinic` = paste(rep(" ", 30), collapse = " "),                        
  check.names = FALSE,
  stringsAsFactors = FALSE
)

# ==============================================================================
# 【核心修复】：物理拆分两行，精准占据中间位置 (完美避免拉伸)
# ==============================================================================

# 替换下划线为空格
plot_data$`Disease Type` <- gsub("_", " ", plot_data$`Disease Type`)

# 1. 按照疾病类别生成交替的背景颜色
category_factor <- as.numeric(factor(plot_data$`Disease Type`, levels = unique(plot_data$`Disease Type`)))
bg_colors <- ifelse(category_factor %% 2 == 1, "#f4f4f4", "white") 

# 2. 将长文本拆分到多个“物理行”中
plot_data$Display_Type <- ""

for (cat in unique(plot_data$`Disease Type`)) {
  # 找到当前疾病大类对应的所有行索引
  idx <- which(plot_data$`Disease Type` == cat)
  n_rows_cat <- length(idx)
  
  if (cat == "Haematological or immunological") {
    if (n_rows_cat >= 2) {
      # 【关键修改】：只拆分成两行，放在中间的两行
      # 对于 4 行疾病，它会精确计算落在第 2 和第 3 行
      start_pos <- idx[1] + floor((n_rows_cat - 2) / 2) 
      plot_data$Display_Type[start_pos] <- "Haematological or"
      plot_data$Display_Type[start_pos + 1] <- "immunological"
    } else {
      plot_data$Display_Type[idx[1]] <- cat
    }
  } else {
    # 常规处理：其他短名称直接放在该组的中间行
    mid_pos <- idx[ceiling(n_rows_cat / 2)]
    plot_data$Display_Type[mid_pos] <- cat
  }
}

# 3. 覆盖原列并清理辅助列
plot_data$`Disease Type` <- plot_data$Display_Type
plot_data$Display_Type <- NULL

# ==============================================================================
# 5. 准备森林图的估计值和置信区间数据
# ==============================================================================

# 删除第3个模型的估计值
est_list <- list(
  -combined_data$diff_mymodel_base_mean,
  -combined_data$diff_mymodel_traits_mean
)

lower_list <- list(
  -combined_data$diff_mymodel_base_ci_upper,
  -combined_data$diff_mymodel_traits_ci_upper
)

upper_list <- list(
  -combined_data$diff_mymodel_base_ci_lower,
  -combined_data$diff_mymodel_traits_ci_lower
)

# ==============================================================================
# 6. 设置视觉风格 - 【保留行高与字体设置】
# ==============================================================================

# 动态获取行数和列数（现在是5列）
n_rows <- nrow(plot_data)
n_cols <- ncol(plot_data)

# 构建和表格尺寸完全一致的对齐矩阵 (默认全是 0 左对齐)
align_hjust <- matrix(0, nrow = n_rows, ncol = n_cols)
align_x <- matrix(0, nrow = n_rows, ncol = n_cols)

# 仅针对第 3 列（C-index列）设置为 0.5 居中对齐
align_hjust[, 3] <- 0.5
align_x[, 3] <- 0.5

tm <- forest_theme(
  base_size = 14,  
  base_family = "sans",
  legend_value = c("Base", "Clinic"), # 移除最后一个图例
  legend_gp = gpar(fontsize = 14, fontface = "bold", cex = 1.0), 
  legend_position = "bottom",
  ci_pch = c(15, 15), # 移除最后一个图例标记形状
  ci_col = c("#377eb8", "#4daf4a"), # 移除最后一个颜色
  ci_lwd = 1.5,
  ci_alpha = 0.9,
  ci_Theight = 0.1,  
  refline_gp = gpar(col = "#90EE90", lty = "solid", lwd = 1.5),
  vertline_gp = gpar(lty = "dashed", col = "grey70", lwd = 1),
  
  core = list(
    # 保留舒适的行高 padding 
    padding = unit(c(6, 3, 6, 3), "mm"), 
    colwidths = unit(c(5, 6, 3.5, 2.5, 2.5), "cm"), # 修改为5列对应的宽度，稍微放宽最后两列
    bg_params = list(fill = bg_colors, col = NA),  
    fg_params = list(
      hjust = as.vector(align_hjust),  
      x = as.vector(align_x)
    )
  ),
  
  colhead = list(
    fg_params = list(
      hjust = c(0, 0, 0.5, 0.5, 0.5), # 调整为5个元素的居中属性
      x = c(0, 0, 0.5, 0.5, 0.5),
      fontface = "bold",
      fontsize = 15 
    )
  )
)

# ==============================================================================
# 7. 生成森林图 - 独立 X 轴范围
# ==============================================================================

calculate_xlim <- function(est, lower, upper) {
  all_values <- c(est, lower, upper)
  all_values <- all_values[!is.na(all_values)]
  
  if (length(all_values) > 0) {
    data_min <- min(all_values)
    data_max <- max(all_values)
    
    data_range <- data_max - data_min
    margin <- data_range * 0.15 
    
    xlim_min <- data_min - margin
    xlim_max <- data_max + margin
    
    if (xlim_min > 0) { xlim_min <- -margin }
    if (xlim_max < 0) { xlim_max <- margin }
    
    return(list(min = xlim_min, max = xlim_max))
  } else {
    return(list(min = -0.2, max = 0.2))
  }
}

# 移除第3个列表项计算
xlim_list <- list(
  calculate_xlim(est_list[[1]], lower_list[[1]], upper_list[[1]]),  
  calculate_xlim(est_list[[2]], lower_list[[2]], upper_list[[2]])
)

generate_ticks <- function(xlim_min, xlim_max) {
  data_range_abs <- max(abs(xlim_min), abs(xlim_max))
  
  if (data_range_abs <= 0.05) {
    by_val <- 0.02
  } else if (data_range_abs <= 0.15) {
    by_val <- 0.05
  } else if (data_range_abs <= 0.5) {
    by_val <- 0.1
  } else {
    by_val <- 0.2
  }
  
  ticks_min <- floor(xlim_min / by_val) * by_val
  ticks_max <- ceiling(xlim_max / by_val) * by_val
  ticks_seq <- seq(from = ticks_min, to = ticks_max, by = by_val)
  return(round(ticks_seq, 2))
}

independent_ticks <- lapply(xlim_list, function(x) generate_ticks(x$min, x$max))
independent_xlim <- lapply(xlim_list, function(x) c(x$min, x$max))

p <- forest(
  data = plot_data,
  est = est_list,
  lower = lower_list,
  upper = upper_list,
  ci_column = 4:5, # 绘图列范围从 4:6 改为 4:5
  ref_line = 0.0,
  xlim = independent_xlim,       
  ticks_at = independent_ticks,  
  colgap = unit(4, "mm"),
  theme = tm,
  margin = unit(c(0.5, 0.5, 0.5, 0.5), "cm")  
)

options(repr.plot.width = 18, repr.plot.height = 12)
print(p)
# ==============================================================================
# 8. 保存图片
# ==============================================================================

save_dir <- "/home/xuln/olink_disease_predict/ProPathNet-github/Figure/"
png_file <- paste0(save_dir, "forest_results.png") # 文件名已微调反映列数变化

n_rows <- nrow(plot_data)
# 高度系数设定为 80，保证加了 padding 后不会被截断
png_height <- max(1500, n_rows * 80 + 200 )  
# 删去一列后适当减小了图片宽度，保持比例协调
png_width <- 5200 

png(png_file, width = png_width, height = png_height, res = 300)
grid.draw(p)  
dev.off()

print(paste("\n包含完美对齐、2行物理换行和行高优化的森林图已保存到:", png_file))