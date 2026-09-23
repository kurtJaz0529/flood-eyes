# -*- coding: utf-8 -*-
"""用本机 Word 把两个 docx 转成 PDF"""
import sys, io, os
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8")
import win32com.client

files = [
    r"C:\Users\21679\.kimi-code\sessions\wd_system32_bd23f52a41d5\session_185aff38-865c-4a6a-8013-50cddbb33041\attachments\f_14e19b4f-7698-4179-bf8e-48f78f44b9fb-附件4：《成都理工大学大学生创新训练计划项目申报诚信承诺书》(1).docx",
    r"C:\Users\21679\.kimi-code\sessions\wd_system32_bd23f52a41d5\session_185aff38-865c-4a6a-8013-50cddbb33041\attachments\f_7258c43c-b5dd-4cac-9fbd-2d72b31c855c-附件2：《成都理工大学成都理工大学2026年大学生创新训练计划项目（第二批次）申报书》(1).docx",
]

word = win32com.client.Dispatch("Word.Application")
word.Visible = False
try:
    for f in files:
        out = os.path.splitext(f)[0] + ".pdf"
        doc = word.Documents.Open(f, ReadOnly=True)
        doc.SaveAs(out, FileFormat=17)  # 17 = wdFormatPDF
        doc.Close(False)
        print("OK:", out, os.path.getsize(out) // 1024, "KB")
finally:
    word.Quit()
