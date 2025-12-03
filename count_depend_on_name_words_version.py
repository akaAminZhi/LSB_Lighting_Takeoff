import fitz  # PyMuPDF
import re
from collections import Counter
import pandas as pd

# 打开 PDF 文件
file_name = "Architectural_L0"
doc = fitz.open(f"pdf_files\{file_name}.pdf")

# 定义正则规则
pattern = r"^(?=.{2,}$)(?:(?=.{1,3}$)[F][A-Z0-9-]*|(?=.*\d)[F][A-Z0-9-]*)$"
# pattern = r"^(?=.{2,}$)(?:(?=.{1,3}$)[BSXF][A-Z0-9-]*|(?=.*\d)[BSXF][A-Z0-9-]*)$"

# for architecture
# pattern = r"^(?=.{2,}$)(?:(?=.{1,3}$)[F][A-Z0-9-]*|(?=.*\d)[F][A-Z0-9-]*)$"


# 输出 Excel 文件路径
output_path = f"words_count\{file_name}_count_result.xlsx"

total_counter = Counter()
# 使用 ExcelWriter 以便写多个 sheet
with pd.ExcelWriter(output_path, engine="xlsxwriter") as writer:
    # 遍历每一页
    for i, page in enumerate(doc):
        words = page.get_text("words")

        # 只取出文本部分（第5个元素）
        word_list = [w[4] for w in words if w[4].strip()]

        # 正则匹配
        matches = [w for w in word_list if re.fullmatch(pattern, w)]

        counter = Counter(matches)
        total_counter.update(counter)
        # 转为 DataFrame
        df = pd.DataFrame(counter.items(), columns=["Word", "Count"]).sort_values(
            by=["Count", "Word"], ascending=[False, True]
        )

        # 写入单独 sheet
        sheet_name = f"Page_{i+1}"
        df.to_excel(writer, index=False, sheet_name=sheet_name)

        print(f"✅ Page {i+1} processed, {len(df)} unique words found.")
    df_total = pd.DataFrame(
        total_counter.items(), columns=["Word", "Count"]
    ).sort_values(by=["Count", "Word"], ascending=[False, True])
    sheet_name = f"total"
    df_total.to_excel(writer, index=False, sheet_name=sheet_name)


print(f"\n🎉 所有页面统计结果已保存到 {output_path}")
