"""
title: Universal File Generator
author: AI Assistant
version: 2.0.0
requirements: fastapi, python-docx, pandas, openpyxl, reportlab
description: Generate various file formats and provide download via Data URI (no internal API dependency)
"""

import csv
import json
import xml.etree.ElementTree as ET
import io
import zipfile
import base64
from typing import Dict, List, Any, Optional, Union
from datetime import datetime
from pydantic import BaseModel, Field

# Optional dependencies with graceful fallback
try:
    from docx import Document
    from docx.shared import Inches

    DOCX_AVAILABLE = True
except ImportError:
    DOCX_AVAILABLE = False

try:
    import pandas as pd

    PANDAS_AVAILABLE = True
except ImportError:
    PANDAS_AVAILABLE = False

try:
    from reportlab.pdfgen import canvas
    from reportlab.lib.pagesizes import letter
    from reportlab.lib.styles import getSampleStyleSheet
    from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer

    PDF_AVAILABLE = True
except ImportError:
    PDF_AVAILABLE = False


class FileGenerator:
    """Internal file generation engine - not exposed to AI"""

    def __init__(self):
        self.mime_types = {
            "csv": "text/csv",
            "json": "application/json",
            "xml": "application/xml",
            "txt": "text/plain",
            "html": "text/html",
            "md": "text/markdown",
            "docx": "application/vnd.openxmlformats-officedocument.wordprocessingml.document",
            "pdf": "application/pdf",
            "xlsx": "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
            "zip": "application/zip",
            "js": "text/javascript",
            "py": "text/x-python",
            "sql": "application/sql",
        }

    def generate_content(self, file_type: str, data: Any, **kwargs) -> Optional[bytes]:
        """Generate file content based on type"""

        try:
            if file_type == "csv":
                return self.generate_csv(data, **kwargs)
            elif file_type == "json":
                return self.generate_json(data, **kwargs)
            elif file_type == "xml":
                return self.generate_xml(data, **kwargs)
            elif file_type == "txt":
                return self.generate_txt(data, **kwargs)
            elif file_type == "html":
                return self.generate_html(data, **kwargs)
            elif file_type == "md":
                return self.generate_markdown(data, **kwargs)
            elif file_type == "docx":
                return self.generate_docx(data, **kwargs)
            elif file_type == "pdf":
                return self.generate_pdf(data, **kwargs)
            elif file_type == "xlsx":
                return self.generate_xlsx(data, **kwargs)
            elif file_type == "zip":
                return self.generate_zip(data, **kwargs)
            elif file_type == "js":
                return self.generate_javascript(data, **kwargs)
            elif file_type == "py":
                return self.generate_python(data, **kwargs)
            elif file_type == "sql":
                return self.generate_sql(data, **kwargs)
            else:
                return None
        except Exception as e:
            raise Exception(f"Content generation failed: {str(e)}")

    def get_mime_type(self, file_type: str) -> str:
        """Get MIME type for file format"""
        return self.mime_types.get(file_type, "application/octet-stream")

    def generate_csv(self, data: Union[List[Dict], List[List], str], **kwargs) -> bytes:
        """Generate CSV content"""
        if not data:
            raise ValueError("No data provided for CSV generation")

        # DEBUG: Let's see what we're actually getting
        data_type = type(data).__name__
        data_preview = str(data)[:100] if len(str(data)) > 100 else str(data)
        print(f"[DEBUG] CSV Generator - Data type: {data_type}")
        print(f"[DEBUG] CSV Generator - Data preview: {repr(data_preview)}")

        # IMPORTANT: Check for string FIRST before any other operations
        if isinstance(data, str):
            print(f"[DEBUG] Detected string data, returning as-is")
            return data.encode("utf-8")

        # If data is not a list, try to convert it
        if not isinstance(data, list):
            raise ValueError(
                f"CSV data must be a list of dictionaries, list of lists, or CSV string. Got: {type(data)}"
            )

        # Check if list is empty
        if len(data) == 0:
            raise ValueError("CSV data list is empty")

        print(
            f"[DEBUG] Processing as structured data - first element type: {type(data[0])}"
        )

        output = io.StringIO()

        if isinstance(data[0], dict):
            # List of dictionaries
            print(f"[DEBUG] Processing as dict list")
            writer = csv.DictWriter(output, fieldnames=data[0].keys())
            writer.writeheader()
            writer.writerows(data)
        elif isinstance(data[0], (list, tuple)):
            # List of lists/tuples
            print(f"[DEBUG] Processing as list of lists")
            writer = csv.writer(output)
            writer.writerows(data)
        else:
            # Fallback: treat as simple values
            print(
                f"[DEBUG] Processing as simple values - first element: {repr(data[0])}"
            )
            writer = csv.writer(output)
            writer.writerow(["value"])  # header
            for item in data:
                writer.writerow([item])

        result = output.getvalue()
        print(f"[DEBUG] Generated CSV preview: {repr(result[:100])}")
        return result.encode("utf-8")

    def generate_json(self, data: Any, **kwargs) -> bytes:
        """Generate JSON content"""
        indent = kwargs.get("indent", 2)
        json_str = json.dumps(data, indent=indent, ensure_ascii=False, default=str)
        return json_str.encode("utf-8")

    def generate_xml(self, data: Dict, **kwargs) -> bytes:
        """Generate XML content"""
        root_name = kwargs.get("root_name", "root")

        def dict_to_xml(d, parent):
            if isinstance(d, dict):
                for key, value in d.items():
                    if isinstance(value, list):
                        for item in value:
                            elem = ET.SubElement(parent, key)
                            dict_to_xml(item, elem)
                    else:
                        elem = ET.SubElement(parent, key)
                        dict_to_xml(value, elem)
            else:
                parent.text = str(d)

        root = ET.Element(root_name)
        dict_to_xml(data, root)
        ET.indent(root, space="  ")

        return ET.tostring(root, encoding="utf-8", xml_declaration=True)

    def generate_txt(self, data: Union[str, List[str]], **kwargs) -> bytes:
        """Generate text content"""
        if isinstance(data, list):
            content = "\n".join(str(item) for item in data)
        else:
            content = str(data)
        return content.encode("utf-8")

    def generate_html(self, data: Union[str, Dict], **kwargs) -> bytes:
        """Generate HTML content"""
        if isinstance(data, str):
            html_content = data
        else:
            title = data.get("title", "Generated Document")
            body = data.get("body", data.get("content", ""))
            style = data.get(
                "style", "body { font-family: Arial, sans-serif; margin: 2em; }"
            )

            html_content = f"""<!DOCTYPE html>
<html lang="ja">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>{title}</title>
    <style>{style}</style>
</head>
<body>
    {body}
</body>
</html>"""

        return html_content.encode("utf-8")

    def generate_markdown(self, data: Union[str, Dict], **kwargs) -> bytes:
        """Generate Markdown content"""
        if isinstance(data, str):
            content = data
        else:
            content = ""
            if "title" in data:
                content += f"# {data['title']}\n\n"
            if "sections" in data:
                for section in data["sections"]:
                    content += f"## {section.get('title', '')}\n\n"
                    content += f"{section.get('content', '')}\n\n"
            elif "content" in data:
                content += data["content"]

        return content.encode("utf-8")

    def generate_docx(self, data: Union[str, Dict], **kwargs) -> bytes:
        """Generate DOCX content"""
        if not DOCX_AVAILABLE:
            raise ImportError(
                "python-docx is required for DOCX generation. Install with: pip install python-docx"
            )

        doc = Document()

        if isinstance(data, str):
            doc.add_paragraph(data)
        else:
            if "title" in data:
                doc.add_heading(data["title"], 0)

            if "sections" in data:
                for section in data["sections"]:
                    if "title" in section:
                        doc.add_heading(section["title"], level=1)
                    if "content" in section:
                        doc.add_paragraph(section["content"])
            elif "content" in data:
                doc.add_paragraph(data["content"])

        buffer = io.BytesIO()
        doc.save(buffer)
        buffer.seek(0)
        return buffer.read()

    def generate_pdf(self, data: Union[str, Dict], **kwargs) -> bytes:
        """Generate PDF content"""
        if not PDF_AVAILABLE:
            raise ImportError(
                "reportlab is required for PDF generation. Install with: pip install reportlab"
            )

        buffer = io.BytesIO()
        doc = SimpleDocTemplate(buffer, pagesize=letter)
        styles = getSampleStyleSheet()
        story = []

        if isinstance(data, str):
            story.append(Paragraph(data, styles["Normal"]))
        else:
            if "title" in data:
                story.append(Paragraph(data["title"], styles["Title"]))
                story.append(Spacer(1, 12))

            if "sections" in data:
                for section in data["sections"]:
                    if "title" in section:
                        story.append(Paragraph(section["title"], styles["Heading1"]))
                    if "content" in section:
                        story.append(Paragraph(section["content"], styles["Normal"]))
                        story.append(Spacer(1, 12))
            elif "content" in data:
                story.append(Paragraph(data["content"], styles["Normal"]))

        doc.build(story)
        buffer.seek(0)
        return buffer.read()

    def generate_xlsx(self, data: List[Dict], **kwargs) -> bytes:
        """Generate XLSX content"""
        if not PANDAS_AVAILABLE:
            raise ImportError(
                "pandas and openpyxl are required for XLSX generation. Install with: pip install pandas openpyxl"
            )

        df = pd.DataFrame(data)
        buffer = io.BytesIO()

        with pd.ExcelWriter(buffer, engine="openpyxl") as writer:
            df.to_excel(writer, sheet_name="Sheet1", index=False)

        buffer.seek(0)
        return buffer.read()

    def generate_zip(self, files: Dict[str, Any], **kwargs) -> bytes:
        """Generate ZIP content"""
        buffer = io.BytesIO()

        with zipfile.ZipFile(buffer, "w", zipfile.ZIP_DEFLATED) as zf:
            for filename, content in files.items():
                if isinstance(content, str):
                    zf.writestr(filename, content.encode("utf-8"))
                elif isinstance(content, bytes):
                    zf.writestr(filename, content)
                else:
                    # Try to generate as structured data
                    file_ext = (
                        filename.split(".")[-1].lower() if "." in filename else "txt"
                    )
                    try:
                        file_content = self.generate_content(file_ext, content)
                        if file_content:
                            zf.writestr(filename, file_content)
                    except:
                        # Fallback to string representation
                        zf.writestr(filename, str(content).encode("utf-8"))

        buffer.seek(0)
        return buffer.read()

    def generate_javascript(self, data: Union[str, Dict], **kwargs) -> bytes:
        """Generate JavaScript content"""
        if isinstance(data, str):
            content = data
        else:
            content = f"""// Generated JavaScript file
// Created: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}

{data.get('content', '')}
"""
        return content.encode("utf-8")

    def generate_python(self, data: Union[str, Dict], **kwargs) -> bytes:
        """Generate Python content"""
        if isinstance(data, str):
            content = data
        else:
            content = f"""#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# Generated Python file
# Created: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}

{data.get('content', '')}
"""
        return content.encode("utf-8")

    def generate_sql(self, data: Union[str, Dict], **kwargs) -> bytes:
        """Generate SQL content"""
        if isinstance(data, str):
            content = data
        else:
            content = f"""-- Generated SQL file
-- Created: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}

{data.get('content', '')}
"""
        return content.encode("utf-8")


class Tools:
    """Public API for Open WebUI - using Data URI approach (no internal API dependency)"""

    def __init__(self):
        self.generator = FileGenerator()

    def generate_file(
        self,
        file_type: str = Field(
            ...,
            description="File type: csv, json, xml, txt, html, md, docx, pdf, xlsx, zip, js, py, sql",
        ),
        data: Any = Field(..., description="Data to convert to file format"),
        filename: Optional[str] = Field(None, description="Custom filename (optional)"),
        indent: Optional[int] = Field(
            2, description="JSON indentation (for JSON files)"
        ),
        root_name: Optional[str] = Field(
            "root", description="Root element name (for XML files)"
        ),
        __request__: object = None,
        __user__: dict = {},
    ) -> str:
        """
        Generate a file of specified type from provided data
        Returns an HTML download button that works in the browser

        :param file_type: Type of file to generate
        :param data: Data to convert (format varies by file type)
        :param filename: Optional custom filename
        :param indent: JSON indentation level (default: 2)
        :param root_name: XML root element name (default: 'root')
        :return: HTML with download button
        """

        try:
            # Validate file type
            supported_types = [
                "csv",
                "json",
                "xml",
                "txt",
                "html",
                "md",
                "docx",
                "pdf",
                "xlsx",
                "zip",
                "js",
                "py",
                "sql",
            ]
            if file_type not in supported_types:
                return f"❌ Unsupported file type: {file_type}. Supported: {', '.join(supported_types)}"

            # Generate filename if not provided
            if not filename:
                timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                filename = f"generated_{timestamp}.{file_type}"
            elif not filename.endswith(f".{file_type}"):
                filename += f".{file_type}"

            # Prepare kwargs for file generators
            kwargs = {"indent": indent, "root_name": root_name}

            # Generate file content
            try:
                file_content = self.generator.generate_content(
                    file_type, data, **kwargs
                )

                if file_content is None:
                    return f"❌ Failed to generate {file_type} content"

                if len(file_content) == 0:
                    return f"❌ Generated empty {file_type} content"

            except Exception as e:
                return f"❌ Content generation error: {str(e)}"

            # Create Data URI
            try:
                # Base64 encode the file content
                encoded_content = base64.b64encode(file_content).decode("utf-8")
                mime_type = self.generator.get_mime_type(file_type)
                data_uri = f"data:{mime_type};base64,{encoded_content}"

                # Create HTML download button
                html_response = f"""
✅ **{file_type.upper()} file '{filename}' generated successfully!**

📁 **Size:** {len(file_content)} bytes
🕒 **Created:** {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}

**🔽 Click the button below to download:**

<div style="margin: 10px 0;">
    <a href="{data_uri}" 
       download="{filename}"
       style="
           display: inline-block;
           background-color: #007bff;
           color: white;
           padding: 10px 20px;
           text-decoration: none;
           border-radius: 5px;
           font-weight: bold;
           border: none;
           cursor: pointer;
       ">
        📥 Download {filename}
    </a>
</div>

<details>
<summary>🔍 **Preview** (first 500 characters)</summary>

```{file_type}
{file_content.decode('utf-8', errors='ignore')[:500]}{'...' if len(file_content) > 500 else ''}
```
</details>

<small>💡 **Tip:** The file is generated as a Data URI, so it works directly in your browser without server dependencies!</small>
"""

                return html_response

            except Exception as e:
                return f"❌ Data URI creation failed: {str(e)}"

        except Exception as e:
            return f"❌ Unexpected error generating {file_type} file: {str(e)}"

    def list_supported_formats(
        self, __request__: object = None, __user__: dict = {}
    ) -> str:
        """
        List all supported file formats and their requirements

        :return: List of supported formats with availability status
        """

        formats_info = {
            "csv": "✅ Always available",
            "json": "✅ Always available",
            "xml": "✅ Always available",
            "txt": "✅ Always available",
            "html": "✅ Always available",
            "md": "✅ Always available",
            "js": "✅ Always available",
            "py": "✅ Always available",
            "sql": "✅ Always available",
            "zip": "✅ Always available",
            "docx": (
                "✅ Available"
                if DOCX_AVAILABLE
                else "❌ Requires: pip install python-docx"
            ),
            "pdf": (
                "✅ Available"
                if PDF_AVAILABLE
                else "❌ Requires: pip install reportlab"
            ),
            "xlsx": (
                "✅ Available"
                if PANDAS_AVAILABLE
                else "❌ Requires: pip install pandas openpyxl"
            ),
        }

        result = "📋 **Universal File Generator v2.0 - Supported Formats:**\n\n"
        for fmt, status in formats_info.items():
            result += f"• **{fmt.upper()}**: {status}\n"

        result += f"\n💡 **Usage example:**\n"
        result += f"```\n"
        result += f"generate_file(\n"
        result += f"  file_type='csv',\n"
        result += (
            f"  data=[{{'name': 'Alice', 'age': 25}}, {{'name': 'Bob', 'age': 30}}],\n"
        )
        result += f"  filename='users.csv'\n"
        result += f")\n"
        result += f"```\n\n"
        result += f"🎯 **New in v2.0:** Uses Data URI approach - no server API dependencies!\n"
        result += f"📱 **Browser compatible:** Works in all modern browsers\n"
        result += f"🔒 **Privacy friendly:** No file upload to servers required"

        return result
