"""
title: Universal File Generator (Pandoc Edition)
author: Skyzi000 & Claude
version: 0.20.0-pandoc
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
                        except:
                            zf.writestr(f"{filename}.txt", f"Failed to download: {content}")
                    else:
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
                            zf.writestr(path, content.encode('utf-8'))
                        else:
                            zf.writestr(path, str(content).encode('utf-8'))
                    elif 'url' in item:
                        # Download from URL
                        try:
                            response = requests.get(item['url'], timeout=10)
                            if response.status_code == 200:
                                zf.writestr(path, response.content)
                        except:
                            zf.writestr(f"{path}.txt", f"Failed to download: {item['url']}")


# Upload services configuration (same as original)
UPLOAD_SERVICES = [
    {
        "name": "transfer.sh",
        "url": "https://transfer.sh",
        "method": "put",
        "retention": "14 days"
    },
    {
        "name": "0x0.st",
        "url": "https://0x0.st",
        "method": "post",
        "retention": "30 days to 1 year (depends on file size)"
    },
    {
        "name": "file.io",
        "url": "https://file.io",
        "method": "post_with_form",
        "retention": "1 download or 14 days"
    },
    {
        "name": "litterbox",
        "url": "https://litterbox.catbox.moe/resources/internals/api.php",
        "method": "litterbox",
        "retention": "1 hour (temporary file hosting)"
    }
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
    
    # Try multiple services in order
    services = [
        {
            "name": "transfer.sh",
            "url": f"https://transfer.sh/{safe_filename}",
            "method": "put",
            "retention": "Depends on service settings"
        },
        {
            "name": "0x0.st", 
            "url": "https://0x0.st",
            "method": "post",
            "retention": "30 days to 1 year (depends on file size)"
        },
        {
            "name": "file.io",
            "url": "https://file.io",
            "method": "post", 
            "retention": "14 days (deleted after first download)"
        },
        {
            "name": "litterbox",
            "url": "https://litterbox.catbox.moe/resources/internals/api.php",
            "method": "litterbox",
            "retention": "1 hour (temporary file hosting)"
        }
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
                response = requests.put(service["url"], data=file_content, headers=headers_with_content_type, timeout=15)
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
                    # transfer.sh and 0x0.st return plain text URL
                    download_url = response.text.strip()
                    
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
    
    return f"""## ❌ Upload Failed on All Services

**File:** `{filename}` ({file_size} bytes)

### Debug Information
```
{error_details}
```

Attempted services: transfer.sh, 0x0.st, file.io, litterbox

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
        data: Any = Field(..., description="Data to convert - STRING format preferred for DOCX/PDF (HTML/Markdown/text). Use Dict only for XLSX/ZIP."),
        filename: Optional[str] = Field(None, description="Custom filename (optional)"),
        password: Optional[str] = Field(None, description="Password for ZIP encryption (optional)"),
        __request__: Optional[Request] = None,
        __user__: Optional[BaseModel] = None,
        __event_emitter__: Optional[Callable[[dict], Awaitable[None]]] = None
    ) -> str:
        """
        Generate a file using Pandoc for superior document conversion
        
        :param file_type: File extension (e.g., 'docx', 'pdf', 'zip')
        :param data: Data to convert - STRING format preferred for DOCX/PDF
        :param filename: Optional custom filename
        :param password: Optional password for ZIP encryption
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