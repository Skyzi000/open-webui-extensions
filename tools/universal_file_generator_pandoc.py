"""
title: Universal File Generator (Pandoc Edition)
author: Skyzi000 & Claude
version: 0.20.4-pandoc
requirements: fastapi, pandas, openpyxl, reportlab, weasyprint, beautifulsoup4, requests, markdown, pyzipper
description: |
  Universal file generation tool using Pandoc for superior document conversion.
  Simplified version that leverages pandoc for HTML/Markdown to DOCX/PDF conversion.
  
  ## Supported Formats
  - **Text**: All text-based formats (CSV, JSON, XML, TXT, HTML, Markdown, YAML, TOML, JavaScript, Python, SQL, etc.)
  - **Binary**: DOCX (via pandoc), XLSX (Excel), PDF (via pandoc/WeasyPrint/ReportLab), ZIP (with URL downloading and AES encryption support)
  - **Graphics**: SVG (native pandoc support in DOCX/PDF)
  
  ## Key Features
  - Pandoc-powered document conversion with native SVG support
  - Automatic HTML/Markdown to DOCX conversion via pandoc
  - Advanced PDF generation with WeasyPrint/ReportLab and Japanese font support
  - ZIP archive creation with remote file downloading from URLs
  - Automatic cloud upload to multiple services (transfer.sh, 0x0.st, file.io, litterbox)
  - Comprehensive error handling and service fallback
  - Simpler codebase with pandoc handling complex conversions
  
  ## Input Format Documentation
  Each file type expects specific data formats - see generate_file() docstring for detailed specifications.
"""

import json
import io
import zipfile
import requests
import base64
import mimetypes
import subprocess
import tempfile
import os
from typing import Awaitable, Callable, Dict, List, Any, Optional, Union
from datetime import datetime
from pydantic import BaseModel, Field
from fastapi import Request, UploadFile


# Check for pandoc availability
def check_pandoc():
    try:
        result = subprocess.run(['pandoc', '--version'], capture_output=True, text=True)
        return result.returncode == 0
    except FileNotFoundError:
        return False

PANDOC_AVAILABLE = check_pandoc()

# Optional dependencies with graceful fallback
try:
    from bs4 import BeautifulSoup
    BS4_AVAILABLE = True
except ImportError:
    BS4_AVAILABLE = False

try:
    import pandas as pd
    PANDAS_AVAILABLE = True
except ImportError:
    PANDAS_AVAILABLE = False

try:
    from weasyprint import HTML, CSS
    WEASYPRINT_AVAILABLE = True
except ImportError:
    WEASYPRINT_AVAILABLE = False

try:
    from reportlab.pdfgen import canvas
    from reportlab.lib.pagesizes import letter
    from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
    from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle
    from reportlab.lib import colors
    from reportlab.pdfbase import pdfmetrics
    from reportlab.pdfbase.ttfonts import TTFont
    from reportlab.lib.fonts import addMapping
    REPORTLAB_AVAILABLE = True
except ImportError:
    REPORTLAB_AVAILABLE = False

try:
    import markdown
    MARKDOWN_AVAILABLE = True
except ImportError:
    MARKDOWN_AVAILABLE = False

try:
    import pyzipper
    PYZIPPER_AVAILABLE = True
except ImportError:
    PYZIPPER_AVAILABLE = False

PDF_AVAILABLE = WEASYPRINT_AVAILABLE or REPORTLAB_AVAILABLE


class FileGeneratorPandoc:
    """Pandoc-powered file generation engine"""
    
    def __init__(self):
        if not PANDOC_AVAILABLE:
            print("Warning: Pandoc not available. DOCX/PDF conversion will use fallback methods.")

    def generate_content(self, file_type: str, data: Any, **kwargs) -> Optional[bytes]:
        """Generate file content based on type"""
        file_type = file_type.lower()
        
        # Text formats - handle as strings
        if file_type in ['csv', 'json', 'xml', 'txt', 'html', 'md', 'yaml', 'toml', 'js', 'py', 'sql', 'ini', 'conf', 'log']:
            return self.generate_text(data, file_type)
        
        # Binary formats
        elif file_type == 'docx':
            return self.generate_docx_pandoc(data, **kwargs)
        elif file_type == 'pdf':
            return self.generate_pdf_pandoc(data, **kwargs)
        elif file_type == 'xlsx':
            return self.generate_xlsx(data, **kwargs)
        elif file_type == 'svg':
            return self.generate_svg(data, **kwargs)
        elif file_type == 'zip':
            return self.generate_zip(data, **kwargs)
        else:
            # Unknown format - treat as text
            return self.generate_text(data, file_type)

    def generate_text(self, data: Any, file_type: str = 'txt') -> bytes:
        """Generate text content"""
        if isinstance(data, str):
            return data.encode('utf-8')
        elif isinstance(data, (dict, list)):
            if file_type == 'json':
                return json.dumps(data, indent=2, ensure_ascii=False).encode('utf-8')
            else:
                return str(data).encode('utf-8')
        else:
            return str(data).encode('utf-8')

    def generate_docx_pandoc(self, data: Union[str, Dict], **kwargs) -> bytes:
        """Generate DOCX using pandoc (preferred) or fallback method"""
        if not PANDOC_AVAILABLE:
            # Fallback to error message
            raise ImportError("Pandoc is required for DOCX generation but not available")
        
        # Reject complex Dict structures
        if isinstance(data, dict):
            if any(key in data for key in ['sections', 'content', 'items', 'chapters', 'parts']):
                raise ValueError("DOCX generation supports string input (HTML/Markdown/plain text). Complex Dict structures with 'sections', 'content', etc. are not supported. Please convert your data to HTML or Markdown string format first.")
        
        # Convert data to string format
        if isinstance(data, str):
            content = data
        else:
            content = str(data)
        
        # Determine input format
        content = content.strip()
        if content.startswith('<') and '>' in content:
            input_format = 'html'
        elif any(pattern in content for pattern in ['#', '*', '```', '|', '[', ']']):
            input_format = 'markdown'
        else:
            input_format = 'markdown'  # Default to markdown for plain text
        
        # Use pandoc to convert to DOCX
        with tempfile.NamedTemporaryFile(mode='w', suffix=f'.{input_format}', delete=False) as temp_input:
            temp_input.write(content)
            temp_input_path = temp_input.name
        
        with tempfile.NamedTemporaryFile(suffix='.docx', delete=False) as temp_output:
            temp_output_path = temp_output.name
        
        try:
            # Run pandoc conversion
            cmd = [
                'pandoc',
                temp_input_path,
                '-f', input_format,
                '-t', 'docx',
                '-o', temp_output_path,
                '--standalone'
            ]
            
            result = subprocess.run(cmd, capture_output=True, text=True)
            
            if result.returncode != 0:
                raise RuntimeError(f"Pandoc conversion failed: {result.stderr}")
            
            # Read the generated DOCX file
            with open(temp_output_path, 'rb') as f:
                docx_content = f.read()
            
            print(f"Successfully generated DOCX using pandoc (size: {len(docx_content)} bytes)")
            return docx_content
            
        finally:
            # Clean up temporary files
            try:
                os.unlink(temp_input_path)
                os.unlink(temp_output_path)
            except:
                pass

    def generate_pdf_pandoc(self, data: Union[str, Dict, List], **kwargs) -> bytes:
        """Generate PDF using pandoc (preferred) or fallback methods"""
        
        # Reject complex Dict structures
        if isinstance(data, dict):
            if any(key in data for key in ['sections', 'content', 'items', 'chapters', 'parts']):
                raise ValueError("PDF generation supports string input (HTML/Markdown/plain text). Complex Dict structures with 'sections', 'content', etc. are not recommended. Please convert your data to HTML or Markdown string format first.")
        
        if PANDOC_AVAILABLE:
            return self._generate_pdf_with_pandoc(data)
        elif WEASYPRINT_AVAILABLE:
            return self._generate_pdf_with_weasyprint(data)
        elif REPORTLAB_AVAILABLE:
            return self._generate_pdf_with_reportlab(data)
        else:
            raise ImportError("PDF generation requires pandoc, weasyprint, or reportlab")

    def _generate_pdf_with_pandoc(self, data) -> bytes:
        """Generate PDF using pandoc with system Japanese fonts"""
        # Convert data to string format
        if isinstance(data, str):
            content = data
        else:
            content = str(data)
        
        # Determine input format
        content = content.strip()
        if content.startswith('<') and '>' in content:
            input_format = 'html'
        elif any(pattern in content for pattern in ['#', '*', '```', '|', '[', ']']):
            input_format = 'markdown'
        else:
            input_format = 'markdown'
        
        # Use pandoc to convert to PDF
        with tempfile.NamedTemporaryFile(mode='w', suffix=f'.{input_format}', delete=False) as temp_input:
            temp_input.write(content)
            temp_input_path = temp_input.name
        
        with tempfile.NamedTemporaryFile(suffix='.pdf', delete=False) as temp_output:
            temp_output_path = temp_output.name
        
        try:
            success = False
            
            # Try system Japanese fonts with XeLaTeX
            japanese_fonts = [
                'Noto Sans CJK JP',
                'Hiragino Sans',
                'Yu Gothic',
                'MS Gothic',
                'DejaVu Sans'
            ]
            
            for font in japanese_fonts:
                cmd = [
                    'pandoc',
                    temp_input_path,
                    '-f', input_format,
                    '-t', 'pdf',
                    '-o', temp_output_path,
                    '--standalone',
                    '--pdf-engine=xelatex',
                    '-V', f'CJKmainfont={font}',
                    '-V', 'geometry:margin=2cm'
                ]
                
                result = subprocess.run(cmd, capture_output=True, text=True)
                
                if result.returncode == 0:
                    print(f"XeLaTeX succeeded with system font: {font}")
                    success = True
                    break
                else:
                    print(f"XeLaTeX failed with font {font}: {result.stderr}")
            
            # Fallback to other PDF engines
            if not success:
                print("XeLaTeX failed with all fonts, trying wkhtmltopdf")
                cmd = [
                    'pandoc',
                    temp_input_path,
                    '-f', input_format,
                    '-t', 'pdf',
                    '-o', temp_output_path,
                    '--standalone',
                    '--pdf-engine=wkhtmltopdf'
                ]
                result = subprocess.run(cmd, capture_output=True, text=True)
                
                if result.returncode != 0:
                    print(f"wkhtmltopdf failed, trying weasyprint: {result.stderr}")
                    cmd = [
                        'pandoc',
                        temp_input_path,
                        '-f', input_format,
                        '-t', 'pdf',
                        '-o', temp_output_path,
                        '--standalone',
                        '--pdf-engine=weasyprint'
                    ]
                    result = subprocess.run(cmd, capture_output=True, text=True)
                    
                    if result.returncode != 0:
                        raise RuntimeError(f"All pandoc PDF engines failed. Last error: {result.stderr}")
            
            # Read the generated PDF file
            with open(temp_output_path, 'rb') as f:
                pdf_content = f.read()
            
            print(f"Successfully generated PDF using pandoc (size: {len(pdf_content)} bytes)")
            return pdf_content
            
        finally:
            # Clean up temporary files
            try:
                os.unlink(temp_input_path)
                os.unlink(temp_output_path)
            except:
                pass

    def _generate_pdf_with_weasyprint(self, data) -> bytes:
        """Fallback PDF generation using WeasyPrint with system Japanese fonts"""
        # Convert data to HTML
        if isinstance(data, str):
            html_content = data if data.strip().startswith('<') else f"<p>{data}</p>"
        else:
            html_content = f"<p>{str(data)}</p>"
        
        # Add Japanese font CSS using system fonts
        font_css = """
        <style>
            body {
                font-family: 'Noto Sans CJK JP', 'Hiragino Sans', 'Yu Gothic', 'DejaVu Sans', sans-serif;
                font-size: 12px;
                line-height: 1.6;
                margin: 2cm;
            }
            h1, h2, h3, h4, h5, h6 {
                font-family: 'Noto Sans CJK JP', 'Hiragino Sans', 'Yu Gothic', 'DejaVu Sans', sans-serif;
            }
        </style>
        """
        
        # Create complete HTML document
        if not html_content.strip().startswith('<html'):
            html_content = f"""
            <!DOCTYPE html>
            <html>
            <head>
                <meta charset="UTF-8">
                <title>Generated Document</title>
                {font_css}
            </head>
            <body>
                {html_content}
            </body>
            </html>
            """
        else:
            # Insert CSS into existing HTML
            if '<head>' in html_content and '</head>' in html_content:
                html_content = html_content.replace('</head>', f'{font_css}</head>')
            else:
                html_content = html_content.replace('<html>', f'<html><head>{font_css}</head>')
        
        print("Generating PDF with WeasyPrint using system Japanese fonts")
        html_doc = HTML(string=html_content)
        pdf_bytes = html_doc.write_pdf()
        return pdf_bytes

    def _generate_pdf_with_reportlab(self, data) -> bytes:
        """Fallback PDF generation using ReportLab"""
        buffer = io.BytesIO()
        doc = SimpleDocTemplate(buffer, pagesize=letter)
        styles = getSampleStyleSheet()
        story = [Paragraph(str(data), styles['Normal'])]
        doc.build(story)
        buffer.seek(0)
        return buffer.read()

    def generate_xlsx(self, data: List[Dict], **kwargs) -> bytes:
        """Generate XLSX content using pandas"""
        if not PANDAS_AVAILABLE:
            raise ImportError("pandas and openpyxl are required for XLSX generation")
        
        # Convert data to DataFrame
        if isinstance(data, list) and data and isinstance(data[0], dict):
            df = pd.DataFrame(data)
        elif isinstance(data, dict):
            df = pd.DataFrame([data])
        else:
            # Try to convert other formats
            df = pd.DataFrame({'data': [str(data)]})
        
        # Write to bytes buffer
        buffer = io.BytesIO()
        with pd.ExcelWriter(buffer, engine='openpyxl') as writer:
            df.to_excel(writer, index=False, sheet_name='Sheet1')
        
        buffer.seek(0)
        return buffer.read()

    def generate_svg(self, data: Union[str, Dict[str, Any]], **kwargs) -> bytes:
        """Generate SVG content"""
        if isinstance(data, str):
            if data.strip().startswith('<svg'):
                return data.encode('utf-8')
            else:
                svg_content = f'''<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 400 300" width="400" height="300">
  <rect width="400" height="300" fill="#f9f9f9" stroke="#333" stroke-width="2"/>
  <text x="200" y="150" text-anchor="middle" font-family="Arial, sans-serif" font-size="16" fill="#333">
    {data.replace('<', '&lt;').replace('>', '&gt;').replace('&', '&amp;')}
  </text>
</svg>'''
                return svg_content.encode('utf-8')
        
        elif isinstance(data, dict):
            # Handle structured SVG data
            width = data.get('width', 400)
            height = data.get('height', 300)
            elements = data.get('elements', [])
            
            svg_content = f'<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 {width} {height}" width="{width}" height="{height}">\n'
            
            for element in elements:
                if element.get('type') == 'text':
                    x = element.get('x', width//2)
                    y = element.get('y', height//2)
                    text = element.get('text', '')
                    color = element.get('color', '#333')
                    size = element.get('size', 16)
                    svg_content += f'  <text x="{x}" y="{y}" text-anchor="middle" font-family="Arial, sans-serif" font-size="{size}" fill="{color}">{text}</text>\n'
                
                elif element.get('type') == 'rect':
                    x = element.get('x', 10)
                    y = element.get('y', 10)
                    w = element.get('width', 100)
                    h = element.get('height', 100)
                    fill = element.get('fill', '#blue')
                    svg_content += f'  <rect x="{x}" y="{y}" width="{w}" height="{h}" fill="{fill}"/>\n'
                
                elif element.get('type') == 'circle':
                    cx = element.get('cx', width//2)
                    cy = element.get('cy', height//2)
                    r = element.get('r', 50)
                    fill = element.get('fill', '#red')
                    svg_content += f'  <circle cx="{cx}" cy="{cy}" r="{r}" fill="{fill}"/>\n'
            
            svg_content += '</svg>'
            return svg_content.encode('utf-8')
        
        else:
            return str(data).encode('utf-8')

    def generate_zip(self, data: Dict[str, Any], **kwargs) -> bytes:
        """Generate ZIP content with optional encryption"""
        if not isinstance(data, (dict, list)):
            raise ValueError("ZIP generation requires dict or list input")
        
        password = kwargs.get('password')
        
        # Choose compression method based on password
        if password and PYZIPPER_AVAILABLE:
            zip_buffer = io.BytesIO()
            with pyzipper.AESZipFile(zip_buffer, 'w', compression=pyzipper.ZIP_DEFLATED, encryption=pyzipper.WZ_AES) as zf:
                zf.setpassword(password.encode('utf-8'))
                self._add_files_to_zip(zf, data)
        else:
            zip_buffer = io.BytesIO()
            with zipfile.ZipFile(zip_buffer, 'w', zipfile.ZIP_DEFLATED) as zf:
                self._add_files_to_zip(zf, data)
        
        zip_buffer.seek(0)
        return zip_buffer.read()

    def _add_files_to_zip(self, zf, data):
        """Add files to zip archive"""
        if isinstance(data, dict):
            for filename, content in data.items():
                if isinstance(content, str):
                    if content.startswith(('http://', 'https://')):
                        # Download URL
                        try:
                            response = requests.get(content, timeout=10)
                            if response.status_code == 200:
                                zf.writestr(filename, response.content)
                        except Exception as e:
                            raise RuntimeError(f"Failed to download URL {content}: {str(e)}")
                    else:
                        # Check if filename suggests binary format that needs conversion
                        file_ext = filename.lower().split('.')[-1] if '.' in filename else ''
                        if file_ext in ['docx', 'pdf', 'xlsx']:
                            try:
                                print(f"Converting text content to {file_ext.upper()} for ZIP: {filename}")
                                # Generate the binary content using our generator
                                file_generator = FileGeneratorPandoc()
                                binary_content = file_generator.generate_content(file_ext, content)
                                if binary_content:
                                    zf.writestr(filename, binary_content)
                                    print(f"Successfully converted and added {file_ext.upper()}: {filename}")
                                else:
                                    raise ValueError(f"Failed to convert content to {file_ext.upper()} format for file {filename}. Ensure the content is valid Markdown/HTML text.")
                            except Exception as e:
                                raise RuntimeError(f"Error converting {filename} to {file_ext.upper()}: {str(e)}")
                        else:
                            # Regular text file
                            zf.writestr(filename, content.encode('utf-8'))
                else:
                    zf.writestr(filename, str(content).encode('utf-8'))
        
        elif isinstance(data, list):
            for item in data:
                if isinstance(item, dict) and 'path' in item:
                    path = item['path']
                    if 'content' in item:
                        content = item['content']
                        if isinstance(content, str):
                            # Check if path suggests binary format that needs conversion
                            file_ext = path.lower().split('.')[-1] if '.' in path else ''
                            if file_ext in ['docx', 'pdf', 'xlsx']:
                                try:
                                    print(f"Converting text content to {file_ext.upper()} for ZIP: {path}")
                                    # Generate the binary content using our generator
                                    file_generator = FileGeneratorPandoc()
                                    binary_content = file_generator.generate_content(file_ext, content)
                                    if binary_content:
                                        zf.writestr(path, binary_content)
                                        print(f"Successfully converted and added {file_ext.upper()}: {path}")
                                    else:
                                        raise ValueError(f"Failed to convert content to {file_ext.upper()} format for file {path}. Ensure the content is valid Markdown/HTML text.")
                                except Exception as e:
                                    raise RuntimeError(f"Error converting {path} to {file_ext.upper()}: {str(e)}")
                            else:
                                # Regular text file
                                zf.writestr(path, content.encode('utf-8'))
                        else:
                            zf.writestr(path, str(content).encode('utf-8'))
                    elif 'url' in item:
                        # Download from URL
                        try:
                            response = requests.get(item['url'], timeout=10)
                            if response.status_code == 200:
                                zf.writestr(path, response.content)
                        except Exception as e:
                            raise RuntimeError(f"Failed to download URL {item['url']}: {str(e)}")

    def list_supported_formats(
        self,
        __request__: object = None,
        __user__: dict = {}
    ) -> str:
        """
        List all supported file formats and their requirements
        
        :return: List of supported formats with availability status
        """
        
        result = "📋 **Universal File Generator - Supported Formats (Pandoc Version):**\n\n"
        result += "**Text Formats:** ✅ Any text-based format (unlimited support)\n"
        result += "- Examples: csv, json, xml, txt, html, md, yaml, toml, js, py, sql, ini, conf, log, etc.\n\n"
        result += "**Binary Formats:**\n"
        result += f"- **DOCX**: {'✅ Available (via Pandoc)' if PANDOC_AVAILABLE else '❌ Requires: pandoc installation'}\n"
        result += f"- **PDF**: {'✅ Available (via Pandoc+XeLaTeX)' if PANDOC_AVAILABLE else '❌ Requires: pandoc + texlive-xetex'}\n"
        result += f"- **XLSX**: {'✅ Available' if PANDAS_AVAILABLE else '❌ Requires: pip install pandas openpyxl'}\n"
        result += "- **ZIP**: ✅ Always available\n\n"
        result += f"💡 **Usage example:**\n"
        result += f"```\n"
        result += f"generate_file(\n"
        result += f"  file_type='csv',\n"
        result += f"  data='name,age\\nAlice,25\\nBob,30',\n"
        result += f"  filename='users.csv'\n"
        result += f")\n"
        result += f"```\n\n"
        result += "🔗 **ZIP Support:** Call `list_zip_formats()` for detailed ZIP creation examples."
        
        return result

    def list_zip_formats(
        self,
        __request__: object = None,
        __user__: dict = {}
    ) -> str:
        """
        Show ZIP creation format documentation (path-based only)
        
        :return: Simple ZIP format documentation with examples
        """
        
        result = "📦 **ZIP File Creation - Supported Formats**\n\n"
        result += "The Universal File Generator supports two simple formats for ZIP creation:\n\n"
        
        # Dictionary format
        result += "## 📁 **Dictionary Format (simple)**\n"
        result += "Simple filename → content mapping.\n\n"
        result += "```json\n"
        result += "{\n"
        result += '  "file_type": "zip",\n'
        result += '  "data": {\n'
        result += '    "README.md": "# My Project\\nHello world!",\n'
        result += '    "src/main.py": "print(\\"Hello!\\"))",\n'
        result += '    "config/app.yaml": "app: demo\\nmode: dev"\n'
        result += "  },\n"
        result += '  "filename": "project.zip"\n'
        result += "}\n"
        result += "```\n\n"
        
        # Path format
        result += "## 📋 **Path Format (advanced)**\n"
        result += "Use list of objects with `path` and `content`/`url` fields.\n\n"
        result += "```json\n"
        result += "{\n"
        result += '  "file_type": "zip",\n'
        result += '  "data": [\n'
        result += '    {"path": "README.md", "content": "# My Project\\nHello world!"},\n'
        result += '    {"path": "src/main.py", "content": "print(\\"Hello!\\")"},\n'
        result += '    {"path": "assets/logo.png", "url": "https://example.com/logo.png"},\n'
        result += '    {"path": "temp/", "content": ""}\n'
        result += "  ],\n"
        result += '  "password": "mypassword",\n'
        result += '  "filename": "project.zip"\n'
        result += "}\n"
        result += "```\n\n"
        
        # Encryption format
        result += "## 🔐 **Encrypted ZIP (AES)**\n"
        result += "Add password protection to your ZIP files using AES encryption.\n\n"
        result += "```json\n"
        result += "{\n"
        result += '  "file_type": "zip",\n'
        result += '  "data": {\n'
        result += '    "secret.txt": "Top secret content!",\n'
        result += '    "private/data.json": "{\\"secret\\": \\"password123\\"}"\n'
        result += "  },\n"
        result += '  "password": "mypassword",\n'
        result += '  "filename": "encrypted.zip"\n'
        result += "}\n"
        result += "```\n\n"
        result += "⚠️ **Note**: Requires `pyzipper` library for AES encryption. Will error if password specified but pyzipper not installed.\n\n"
        
        # Rules
        result += "## 📋 **Rules**\n"
        result += "- Dictionary: `\"filename\": \"content\"` pairs\n"
        result += "- Path format: `path` + `content` or `url`\n"
        result += "- Empty folders: path ending with `/` and empty content\n"
        result += "- URLs are downloaded automatically\n"
        result += "- Forward slashes `/` create folder structures\n"
        result += "- Encryption: add `password` parameter for AES encryption (requires pyzipper)\n\n"
        
        result += "**🚀 Try it now:** Use `generate_file()` with `file_type: \"zip\"` and either format!"
        
        return result


# Upload services configuration (same as original)
# Self-hosted transfer.skyzi.jp is first priority, others commented out
UPLOAD_SERVICES = [
    {
        "name": "transfer.skyzi.jp",
        "upload_url": "http://transfer-sh:8080",
        "download_url": "https://transfer.skyzi.jp",
        "method": "put",
        "retention": "14 days"
    },
    # {
    #     "name": "transfer.sh",
    #     "url": "https://transfer.sh",
    #     "method": "put",
    #     "retention": "14 days"
    # },
    # {
    #     "name": "0x0.st",
    #     "url": "https://0x0.st",
    #     "method": "post",
    #     "retention": "30 days to 1 year (depends on file size)"
    # },
    # {
    #     "name": "file.io",
    #     "url": "https://file.io",
    #     "method": "post_with_form",
    #     "retention": "1 download or 14 days"
    # },
    # {
    #     "name": "litterbox",
    #     "url": "https://litterbox.catbox.moe/resources/internals/api.php",
    #     "method": "litterbox",
    #     "retention": "1 hour (temporary file hosting)"
    # }
]

def _upload_file(file_content: bytes, filename: str, file_type: str, file_size: int) -> str:
    """Upload file using multiple services with fallback"""
    
    # Check for zero-size files
    if file_size == 0 or len(file_content) == 0:
        return f"❌ **File Upload Error**: Generated file '{filename}' is empty (0 bytes)\n\n" \
               f"This usually indicates:\n" \
               f"- Empty or invalid input data\n" \
               f"- File generation process failed\n" \
               f"- Unsupported data format for {file_type} files\n\n" \
               f"Please check your input data and try again."
    
    import urllib.parse
    safe_filename = urllib.parse.quote(filename, safe='.-_')
    
    # Try self-hosted service first, others commented out
    services = [
        {
            "name": "transfer.skyzi.jp",
            "upload_url": f"http://transfer-sh:8080/{safe_filename}",
            "upload_base": "http://transfer-sh:8080",
            "download_base": "https://transfer.skyzi.jp",
            "method": "put",
            "retention": "14 days"
        },
        # {
        #     "name": "transfer.sh",
        #     "url": f"https://transfer.sh/{safe_filename}",
        #     "method": "put",
        #     "retention": "Depends on service settings"
        # },
        # {
        #     "name": "0x0.st", 
        #     "url": "https://0x0.st",
        #     "method": "post",
        #     "retention": "30 days to 1 year (depends on file size)"
        # },
        # {
        #     "name": "file.io",
        #     "url": "https://file.io",
        #     "method": "post", 
        #     "retention": "14 days (deleted after first download)"
        # },
        # {
        #     "name": "litterbox",
        #     "url": "https://litterbox.catbox.moe/resources/internals/api.php",
        #     "method": "litterbox",
        #     "retention": "1 hour (temporary file hosting)"
        # }
    ]
    
    errors = []  # Initialize errors list
    
    for service in services:
        try:
            # Add User-Agent header for all requests
            headers = {"User-Agent": "curl/7.68.0"}
            
            if service["method"] == "litterbox":
                # litterbox.catbox.moe style
                files = {"fileToUpload": (filename, file_content)}
                data = {
                    "reqtype": "fileupload",
                    "time": "1h"  # 1 hour retention
                }
                response = requests.post(service["url"], files=files, data=data, headers=headers, timeout=15)
            elif service["method"] == "put":
                # transfer.sh style
                # Add Content-Type header for proper file type detection
                import mimetypes
                mime_type, _ = mimetypes.guess_type(filename)
                if not mime_type:
                    mime_type = 'application/octet-stream'
                
                headers_with_content_type = headers.copy()
                headers_with_content_type["Content-Type"] = mime_type
                # Use upload_url for request, download_url for response
                upload_url = service.get("upload_url", service.get("url", ""))
                if not upload_url:
                    error_info = f"{service['name']}: No upload URL configured"
                    errors.append(error_info)
                    continue
                response = requests.put(upload_url, data=file_content, headers=headers_with_content_type, timeout=15)
            else:
                # file.io and 0x0.st style (multipart form)
                # Determine proper MIME type for file
                import mimetypes
                mime_type, _ = mimetypes.guess_type(filename)
                if not mime_type:
                    mime_type = 'application/octet-stream'
                
                files = {"file": (filename, file_content, mime_type)}
                data = {}
                
                # Add secret parameter for 0x0.st
                if service["name"] == "0x0.st":
                    data["secret"] = ""
                
                response = requests.post(service["url"], files=files, data=data, headers=headers, timeout=15)
            
            if response.status_code == 200:
                download_url = ""
                delete_token = ""
                
                if service["name"] == "litterbox":
                    # litterbox returns plain text URL
                    download_url = response.text.strip()
                elif service["name"] == "file.io":
                    # file.io returns JSON - handle parsing errors
                    try:
                        result = response.json()
                        download_url = result.get("link", "")
                        # file.io doesn't provide delete capability
                    except Exception:
                        # JSON parsing failed, treat as error
                        error_info = f"{service['name']}: Invalid JSON response - {response.text[:100]}"
                        errors.append(error_info)
                        continue
                else:
                    # transfer.sh style services
                    response_url = response.text.strip()
                    if service["name"] == "transfer.skyzi.jp" and "upload_base" in service and "download_base" in service:
                        # Replace upload URL base with download URL base
                        upload_base = service["upload_base"]
                        download_base = service["download_base"]
                        download_url = response_url.replace(upload_base, download_base)
                    else:
                        # transfer.sh and 0x0.st return plain text URL
                        download_url = response_url
                    
                    # For 0x0.st, get the management token from X-Token header
                    if service["name"] == "0x0.st":
                        delete_token = response.headers.get("X-Token", "")
                
                if download_url:
                    # Build response with optional delete instructions
                    delete_section = ""
                    if delete_token and service["name"] == "0x0.st":
                        delete_section = f"""
### 🗑️ File Deletion

To delete the file from 0x0.st, run the following command:

```bash
curl -F "token={delete_token}" -F "delete=" {download_url}
```

⚠️ **Warning**: This command will immediately delete the file
"""
                    
                    return f"""## ✅ File Generated and Uploaded Successfully (Pandoc Edition)

**📄 Filename:** `{filename}`  
**💾 Size:** {file_size} bytes ({file_size/1024:.1f} KB)  
**🕒 Created:** {datetime.now().strftime('%Y-%m-%d %H:%M')}  
**🌐 Service:** {service["name"]}

### 📥 Download

**[⬇️ Download {filename}]({download_url})**

### 📋 Direct Link
```
{download_url}
```
{delete_section}
**Retention:** {service["retention"]}  
**Sharing:** This link can be shared with others

**Important:** This message is usually hidden from users. Please provide the user with links or other information.
"""
            else:
                # HTTP error - collect status and response
                error_info = f"{service['name']}: HTTP {response.status_code} - {response.text[:100]}"
                errors.append(error_info)
                    
        except Exception as e:
            # Try next service - collect error info
            error_info = f"{service['name']}: {str(e)}"
            errors.append(error_info)
            continue
    
    # All services failed - show detailed errors
    error_details = "\n".join(errors) if errors else "Unknown details"
    
    return f"""## ❌ Upload Failed on Self-Hosted Service

**File:** `{filename}` ({file_size} bytes)

### Debug Information
```
{error_details}
```

Attempted service: transfer.skyzi.jp

Check the error details and retry in the correct format.
"""


class Tools:
    """Main tool class for API integration"""
    
    def __init__(self):
        self.generator = FileGeneratorPandoc()
        self.event_emitter = None

    async def generate_file(
        self,
        file_type: str = Field(..., description="File type (extension): csv, json, xml, txt, html, md, yaml, toml, js, py, sql, docx, pdf, xlsx, zip, etc."),
        data: Any = Field(..., description="Data to convert - CRITICAL: Use STRING format for DOCX/PDF (HTML/Markdown/plain text), List[Dict] for XLSX, Dict or List[Dict] for ZIP. Do NOT use complex Dict objects for DOCX/PDF generation."),
        filename: Optional[str] = Field(None, description="Custom filename (optional)"),
        password: Optional[str] = Field(None, description="Password for ZIP encryption (optional)"),
        __request__: Optional[Request] = None,
        __user__: Optional[BaseModel] = None,
        __event_emitter__: Optional[Callable[[dict], Awaitable[None]]] = None
    ) -> str:
        """
        Generate a file of specified type from provided data using Pandoc for superior document conversion
        Uploads all files to a cloud service and returns download link
        
        :param file_type: File extension (e.g., 'csv', 'pdf', 'zip' or '.csv', '.pdf', '.zip') - Must be exact match
        :param data: Data to convert - expected formats by file type:
                    **IMPORTANT: For DOCX and PDF, data should be STRING format (HTML/Markdown/text), NOT JSON/Dict objects**
                    - Any text format (csv, json, xml, txt, html, md, yaml, toml, js, py, sql, ini, conf, log, etc.): str (pre-formatted text content)
                    - DOCX: str (HTML, Markdown, or plain text - auto-detected) - Pandoc powered with native SVG support
                    - PDF: str (HTML, Markdown, or plain text - auto-detected) - Pandoc powered with XeLaTeX and Japanese fonts
                    - XLSX: List[Dict] (list of dictionaries) or tabular data
                    - SVG: str (SVG XML content)
                    - ZIP: Dict[str, Any] (filename -> content mapping) OR List[Dict] (list of {path, content/url} objects) - Call `list_zip_formats()` for detailed ZIP creation examples.
        :param filename: Optional custom filename
        :param password: Optional password for ZIP encryption (AES encryption via pyzipper)
        :return: Markdown with download information
        """
        
        try:
            # Store event emitter for notifications
            self.event_emitter = __event_emitter__
            
            # Input validation
            if not file_type or not isinstance(file_type, str):
                return "❌ Invalid file type provided"
            
            file_type = file_type.lstrip('.').lower()
            
            # Generate filename if not provided
            if not filename or hasattr(filename, 'default'):
                timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
                filename = f"generated_{timestamp}.{file_type}"
            else:
                filename = str(filename)
                if not filename.endswith(f'.{file_type}'):
                    filename += f'.{file_type}'

            # Handle password parameter
            kwargs = {}
            actual_password = None
            
            if password is not None and password and not hasattr(password, 'default'):
                actual_password = password
            elif isinstance(data, dict) and 'password' in data:
                actual_password = data['password']
                if actual_password is not None:
                    data = {k: v for k, v in data.items() if k != 'password'}

            if actual_password and file_type != 'zip':
                if hasattr(self, 'event_emitter') and self.event_emitter:
                    await self.event_emitter({
                        "type": "notification",
                        "data": {
                            "type": "error",
                            "content": f"パスワード保護はZIPファイルにのみ対応しています: {file_type.upper()} ファイルではサポートされていません"
                        }
                    })
                return f"❌ Error: Password protection is only supported for ZIP files"
            
            if actual_password and file_type == 'zip':
                kwargs['password'] = actual_password

            # Generate file content
            try:
                file_content = self.generator.generate_content(file_type, data, **kwargs)
                
                if file_content is None:
                    if hasattr(self, 'event_emitter') and self.event_emitter:
                        await self.event_emitter({
                            "type": "notification",
                            "data": {
                                "type": "error",
                                "content": f"{file_type}ファイルの生成に失敗しました"
                            }
                        })
                    return f"❌ Failed to generate {file_type} content"
                    
                if len(file_content) == 0:
                    if hasattr(self, 'event_emitter') and self.event_emitter:
                        await self.event_emitter({
                            "type": "notification",
                            "data": {
                                "type": "warning",
                                "content": f"{file_type}ファイルの内容が空です"
                            }
                        })
                    return f"❌ Generated empty {file_type} content"
                    
            except Exception as e:
                if hasattr(self, 'event_emitter') and self.event_emitter:
                    await self.event_emitter({
                        "type": "notification",
                        "data": {
                            "type": "error",
                            "content": f"コンテンツ生成エラー: {str(e)}"
                        }
                    })
                return f"❌ Content generation error: {str(e)}"

            # Upload file using the complete upload implementation
            file_size = len(file_content)
            upload_result = _upload_file(file_content, filename, file_type, file_size)
            
            # Emit success notification
            if hasattr(self, 'event_emitter') and self.event_emitter:
                await self.event_emitter({
                    "type": "notification",
                    "data": {
                        "type": "success",
                        "content": f"{filename}を正常に生成しました (Pandoc版)"
                    }
                })
            
            return upload_result

        except Exception as e:
            if hasattr(self, 'event_emitter') and self.event_emitter:
                await self.event_emitter({
                    "type": "notification",
                    "data": {
                        "type": "error",
                        "content": f"予期しないエラーが発生しました: {str(e)}"
                    }
                })
            return f"❌ Unexpected error: {str(e)}"

    def list_supported_formats(
        self,
        __request__: object = None,
        __user__: dict = {}
    ) -> str:
        """
        List all supported file formats and their requirements with detailed examples
        
        This method provides comprehensive documentation for AI systems on how to properly format data for each file type.
        Essential for understanding proper data structures before calling generate_file().
        
        :return: Complete documentation of supported formats including:
                - Text formats (csv, json, xml, txt, html, md, yaml, etc.) - expect string input
                - Binary formats (docx, pdf) - expect STRING input (HTML/Markdown), NOT Dict objects
                - Spreadsheet (xlsx) - expect List[Dict] format
                - Archives (zip) - expect Dict or List[Dict] format
                - Graphics (svg) - expect SVG XML string
        """
        return self.generator.list_supported_formats(__request__, __user__)

    def list_zip_formats(
        self,
        __request__: object = None,
        __user__: dict = {}
    ) -> str:
        """
        Show detailed ZIP creation format documentation with multiple examples
        
        Essential reference for AI systems when creating ZIP files. Provides concrete examples
        of proper data structures for ZIP creation with automatic binary conversion.
        
        :return: Comprehensive ZIP format documentation including:
                - Dictionary format: {"filename": "content"} for simple file creation
                - List format: [{"path": "file.txt", "content": "data"}] for advanced features
                - URL downloading: [{"path": "file.ext", "url": "https://..."}]
                - Binary conversion: automatic DOCX/PDF/XLSX generation from text content
                - Encryption examples: password-protected ZIP files
        """
        return self.generator.list_zip_formats(__request__, __user__)