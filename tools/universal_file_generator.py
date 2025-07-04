"""
title: Universal File Generator
author: AI Assistant
version: 4.0.0
requirements: fastapi, python-docx, pandas, openpyxl, reportlab
description: Advanced file generator with flexible data structure parsing and external service upload
"""

import csv
import json
import xml.etree.ElementTree as ET
import io
import zipfile
import requests
import base64
from typing import Dict, List, Any, Optional, Union
from datetime import datetime
from pydantic import BaseModel, Field
from fastapi import UploadFile

# Optional dependencies with graceful fallback
try:
    from docx import Document
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
    from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
    from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer
    from reportlab.pdfbase import pdfmetrics
    from reportlab.pdfbase.ttfonts import TTFont
    from reportlab.lib.fonts import addMapping
    PDF_AVAILABLE = True
except ImportError:
    PDF_AVAILABLE = False


class FileGenerator:
    """Internal file generation engine - not exposed to AI"""
    
    def __init__(self):
        self.mime_types = {
            'csv': 'text/csv',
            'json': 'application/json',
            'xml': 'application/xml',
            'txt': 'text/plain',
            'html': 'text/html',
            'md': 'text/markdown',
            'docx': 'application/vnd.openxmlformats-officedocument.wordprocessingml.document',
            'pdf': 'application/pdf',
            'xlsx': 'application/vnd.openxmlformats-officedocument.spreadsheetml.sheet',
            'zip': 'application/zip',
            'js': 'text/javascript',
            'py': 'text/x-python',
            'sql': 'application/sql',
        }

    def generate_content(self, file_type: str, data: Any, **kwargs) -> Optional[bytes]:
        """Generate file content based on type"""
        
        try:
            if file_type == 'csv':
                return self.generate_csv(data, **kwargs)
            elif file_type == 'json':
                return self.generate_json(data, **kwargs)
            elif file_type == 'xml':
                return self.generate_xml(data, **kwargs)
            elif file_type == 'txt':
                return self.generate_txt(data, **kwargs)
            elif file_type == 'html':
                return self.generate_html(data, **kwargs)
            elif file_type == 'md':
                return self.generate_markdown(data, **kwargs)
            elif file_type == 'docx':
                return self.generate_docx(data, **kwargs)
            elif file_type == 'pdf':
                return self.generate_pdf(data, **kwargs)
            elif file_type == 'xlsx':
                return self.generate_xlsx(data, **kwargs)
            elif file_type == 'zip':
                return self.generate_zip(data, **kwargs)
            elif file_type == 'js':
                return self.generate_javascript(data, **kwargs)
            elif file_type == 'py':
                return self.generate_python(data, **kwargs)
            elif file_type == 'sql':
                return self.generate_sql(data, **kwargs)
            else:
                return None
        except Exception as e:
            raise Exception(f"Content generation failed: {str(e)}")

    def get_mime_type(self, file_type: str) -> str:
        """Get MIME type for file format"""
        return self.mime_types.get(file_type, 'application/octet-stream')

    def generate_csv(self, data: Union[List[Dict], List[List], str], **kwargs) -> bytes:
        """Generate CSV content"""
        if not data:
            raise ValueError("No data provided for CSV generation")
        
        # IMPORTANT: Check for string FIRST before any other operations
        if isinstance(data, str):
            return data.encode('utf-8')
        
        # If data is not a list, try to convert it
        if not isinstance(data, list):
            raise ValueError(f"CSV data must be a list of dictionaries, list of lists, or CSV string. Got: {type(data)}")
        
        # Check if list is empty
        if len(data) == 0:
            raise ValueError("CSV data list is empty")
        
        output = io.StringIO()
        
        if isinstance(data[0], dict):
            # List of dictionaries
            writer = csv.DictWriter(output, fieldnames=data[0].keys())
            writer.writeheader()
            writer.writerows(data)
        elif isinstance(data[0], (list, tuple)):
            # List of lists/tuples
            writer = csv.writer(output)
            writer.writerows(data)
        else:
            # Fallback: treat as simple values
            writer = csv.writer(output)
            writer.writerow(['value'])  # header
            for item in data:
                writer.writerow([item])
        
        return output.getvalue().encode('utf-8')

    def generate_json(self, data: Any, **kwargs) -> bytes:
        """Generate JSON content"""
        indent = kwargs.get('indent', 2)
        json_str = json.dumps(data, indent=indent, ensure_ascii=False, default=str)
        return json_str.encode('utf-8')

    def generate_xml(self, data: Dict, **kwargs) -> bytes:
        """Generate XML content"""
        root_name = kwargs.get('root_name', 'root')
        
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
        
        return ET.tostring(root, encoding='utf-8', xml_declaration=True)

    def generate_txt(self, data: Union[str, List[str]], **kwargs) -> bytes:
        """Generate text content"""
        if isinstance(data, list):
            content = '\n'.join(str(item) for item in data)
        else:
            content = str(data)
        return content.encode('utf-8')

    def generate_html(self, data: Union[str, Dict], **kwargs) -> bytes:
        """Generate HTML content"""
        if isinstance(data, str):
            html_content = data
        else:
            title = data.get('title', 'Generated Document')
            body = data.get('body', data.get('content', ''))
            style = data.get('style', 'body { font-family: Arial, sans-serif; margin: 2em; }')
            
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
        
        return html_content.encode('utf-8')

    def generate_markdown(self, data: Union[str, Dict], **kwargs) -> bytes:
        """Generate Markdown content"""
        if isinstance(data, str):
            content = data
        else:
            content = ""
            if 'title' in data:
                content += f"# {data['title']}\n\n"
            if 'sections' in data:
                for section in data['sections']:
                    content += f"## {section.get('title', '')}\n\n"
                    content += f"{section.get('content', '')}\n\n"
            elif 'content' in data:
                content += data['content']
        
        return content.encode('utf-8')

    def generate_docx(self, data: Union[str, Dict], **kwargs) -> bytes:
        """Generate DOCX content - handles any AI-generated structure"""
        if not DOCX_AVAILABLE:
            raise ImportError("python-docx is required for DOCX generation. Install with: pip install python-docx")
        
        doc = Document()
        
        if isinstance(data, str):
            self._add_text_content(doc, data)
        else:
            self._parse_any_structure(doc, data)
        
        buffer = io.BytesIO()
        doc.save(buffer)
        buffer.seek(0)
        return buffer.read()
    
    def _parse_any_structure(self, doc, data):
        """Parse any dictionary structure intelligently"""
        if not isinstance(data, dict):
            doc.add_paragraph(str(data))
            return
        
        # Extract title first
        title_keys = ['title', 'タイトル', 'name', 'subject', 'heading']
        for key in title_keys:
            if key in data and data[key]:
                doc.add_heading(str(data[key]), 0)
                break
        
        # Extract metadata 
        meta_keys = ['date', 'author', 'created', 'updated', 'version']
        meta_info = []
        for key in meta_keys:
            if key in data and data[key]:
                meta_info.append(f"{key.title()}: {data[key]}")
        
        if meta_info:
            meta_p = doc.add_paragraph(" | ".join(meta_info))
            meta_p.italic = True
            doc.add_paragraph("")
        
        # Process remaining content
        processed = set(title_keys + meta_keys)
        
        for key, value in data.items():
            if key in processed:
                continue
                
            if isinstance(value, list):
                self._process_list(doc, key, value)
            elif isinstance(value, dict):
                doc.add_heading(str(key).replace('_', ' ').title(), 1)
                self._parse_any_structure(doc, value)
            elif value:
                self._add_key_value(doc, key, value)
    
    def _process_list(self, doc, key, items):
        """Process list items flexibly"""
        if not items:
            return
            
        # Don't add heading for common list names
        if key not in ['sections', 'content', 'items', 'data']:
            doc.add_heading(str(key).replace('_', ' ').title(), 1)
        
        for item in items:
            if isinstance(item, dict):
                self._process_dict_item(doc, item)
            elif isinstance(item, str) and item.strip():
                self._add_text_content(doc, item)
            elif item:
                doc.add_paragraph(str(item))
    
    def _process_dict_item(self, doc, item):
        """Process dictionary item looking for common patterns"""
        heading_keys = ['heading', 'title', 'name', 'section', 'header']
        content_keys = ['body', 'content', 'text', 'description', 'details']
        
        heading = None
        content = None
        
        # Find heading
        for h_key in heading_keys:
            if h_key in item and item[h_key]:
                heading = str(item[h_key])
                break
        
        # Find content  
        for c_key in content_keys:
            if c_key in item and item[c_key]:
                content = str(item[c_key])
                break
        
        if heading:
            doc.add_heading(heading, 2)
        
        if content:
            self._add_text_content(doc, content)
        
        # Process any remaining fields
        processed = set(heading_keys + content_keys)
        for key, value in item.items():
            if key not in processed and value:
                self._add_key_value(doc, key, value)
    
    def _add_key_value(self, doc, key, value):
        """Add key-value pair appropriately"""
        if isinstance(value, str) and len(value) > 50:
            doc.add_heading(str(key).replace('_', ' ').title(), 2)
            self._add_text_content(doc, value)
        else:
            doc.add_paragraph(f"{str(key).replace('_', ' ').title()}: {value}")
    
    def _add_text_content(self, doc, text):
        """Add text with proper line handling"""
        if not text:
            return
            
        # Handle various line break formats
        text = str(text).replace('\\n', '\n')
        lines = text.split('\n')
        
        for line in lines:
            line = line.strip()
            if line:
                doc.add_paragraph(line)

    def generate_pdf(self, data: Union[str, Dict, List], **kwargs) -> bytes:
        """Generate PDF content with flexible data structure support using canvas for Japanese"""
        if not PDF_AVAILABLE:
            raise ImportError("reportlab is required for PDF generation. Install with: pip install reportlab")
        
        # Try canvas approach for better Japanese font control
        return self._generate_pdf_with_canvas(data)
    
    def _generate_pdf_with_canvas(self, data) -> bytes:
        """Generate PDF using canvas for direct font control"""
        buffer = io.BytesIO()
        
        # Force download and register Japanese font with detailed logging
        print("=== PDF Generation with Japanese Font Support ===")
        font_registered = self._download_and_register_japanese_font()
        
        if font_registered:
            font_name = 'NotoSansJP'
            print(f"✓ Using Japanese font: {font_name}")
        else:
            font_name = 'Helvetica'
            print(f"✗ Japanese font failed, using: {font_name}")
        
        # Create canvas
        c = canvas.Canvas(buffer, pagesize=letter)
        width, height = letter
        
        # Set initial position and margins
        y_position = height - 50
        left_margin = 50
        right_margin = 50
        max_width = width - left_margin - right_margin
        line_height = 20
        
        # Convert data to text lines
        text_lines = self._extract_text_lines(data)
        print(f"Extracted {len(text_lines)} lines of text")
        
        for i, line in enumerate(text_lines):
            if y_position < 50:  # New page if needed
                c.showPage()
                y_position = height - 50
            
            # Determine font size based on content
            if line.startswith('# '):
                font_size = 16
                line = line[2:]
                y_position -= 5
            elif line.startswith('## '):
                font_size = 14
                line = line[3:]
                y_position -= 3
            elif line.startswith('• '):
                font_size = 11
            else:
                font_size = 12
            
            # Set font with error handling
            try:
                c.setFont(font_name, font_size)
                print(f"Line {i+1}: Font set to {font_name} {font_size}pt")
            except Exception as font_error:
                print(f"Font setting error: {font_error}")
                c.setFont('Helvetica', font_size)
                font_name = 'Helvetica'  # Fallback permanently
            
            # Draw text with word wrapping
            try:
                # Check if text fits in one line
                text_width = c.stringWidth(line, font_name, font_size)
                
                if text_width <= max_width:
                    # Text fits, draw normally
                    c.drawString(left_margin, y_position, line)
                    print(f"Line {i+1}: Drew text successfully")
                else:
                    # Text too long, need to wrap
                    wrapped_lines = self._wrap_text(line, font_name, font_size, max_width, c)
                    print(f"Line {i+1}: Wrapped into {len(wrapped_lines)} lines")
                    
                    for wrap_line in wrapped_lines:
                        if y_position < 50:  # New page if needed
                            c.showPage()
                            y_position = height - 50
                            c.setFont(font_name, font_size)
                        
                        c.drawString(left_margin, y_position, wrap_line)
                        y_position -= line_height
                    
                    # Skip the normal y_position decrement since we already did it
                    continue
                    
            except Exception as e:
                print(f"Text drawing error for line {i+1}: {e}")
                # Keep original text but add error note
                try:
                    c.drawString(left_margin, y_position, f"[JP Font Error] {line}")
                    print(f"Line {i+1}: Drew with error prefix")
                except:
                    c.drawString(left_margin, y_position, f"[Line {i+1}: Font not available]")
            
            y_position -= line_height
        
        print("PDF generation completed, saving...")
        c.save()
        buffer.seek(0)
        result = buffer.read()
        print(f"PDF saved, size: {len(result)} bytes")
        return result
    
    def _extract_text_lines(self, data) -> list:
        """Extract text lines from various data structures"""
        lines = []
        
        if isinstance(data, str):
            lines = data.split('\n')
        elif isinstance(data, dict):
            # Extract title
            title_keys = ['title', 'タイトル', 'name', 'subject']
            for key in title_keys:
                if key in data and data[key]:
                    lines.append(f"# {data[key]}")
                    lines.append("")
                    break
            
            # Extract content/text
            content_keys = ['content', 'text', 'body', 'description']
            for key in content_keys:
                if key in data and data[key]:
                    if isinstance(data[key], str):
                        lines.extend(data[key].split('\n'))
                    else:
                        lines.append(str(data[key]))
                    lines.append("")
                    break
            
            # Extract sections - with proper field mapping
            if 'sections' in data and isinstance(data['sections'], list):
                for section in data['sections']:
                    if isinstance(section, dict):
                        # Handle heading/title
                        heading_keys = ['heading', 'title', 'name']
                        for key in heading_keys:
                            if key in section and section[key]:
                                lines.append(f"## {section[key]}")
                                break
                        
                        # Handle text/content  
                        text_keys = ['text', 'content', 'body']
                        for key in text_keys:
                            if key in section and section[key]:
                                if isinstance(section[key], str):
                                    lines.extend(section[key].split('\n'))
                                else:
                                    lines.append(str(section[key]))
                                break
                        
                        # Handle list
                        if 'list' in section and isinstance(section['list'], list):
                            for item in section['list']:
                                lines.append(f"• {item}")
                        
                        # Handle table
                        if 'table' in section and isinstance(section['table'], dict):
                            table = section['table']
                            if 'headers' in table and isinstance(table['headers'], list):
                                lines.append(" | ".join(str(h) for h in table['headers']))
                                lines.append("-" * 40)
                            
                            if 'rows' in table and isinstance(table['rows'], list):
                                for row in table['rows']:
                                    if isinstance(row, list):
                                        lines.append(" | ".join(str(cell) for cell in row))
                        
                        lines.append("")
            
            # Extract other fields
            processed = {'title', 'タイトル', 'name', 'subject', 'content', 'text', 'body', 'description', 'sections'}
            for key, value in data.items():
                if key not in processed and value:
                    lines.append(f"## {key}")
                    if isinstance(value, (list, dict)):
                        lines.append(str(value))
                    else:
                        lines.extend(str(value).split('\n'))
                    lines.append("")
        
        elif isinstance(data, list):
            for item in data:
                lines.extend(self._extract_text_lines(item))
        
        return [line.strip() for line in lines if line.strip()]
    
    def _wrap_text(self, text: str, font_name: str, font_size: int, max_width: float, canvas_obj) -> list:
        """Wrap text to fit within specified width"""
        words = text.split(' ')
        lines = []
        current_line = ""
        
        for word in words:
            # Test if adding this word would exceed width
            test_line = current_line + (" " if current_line else "") + word
            test_width = canvas_obj.stringWidth(test_line, font_name, font_size)
            
            if test_width <= max_width:
                current_line = test_line
            else:
                # Current line is full, start new line
                if current_line:
                    lines.append(current_line)
                    current_line = word
                else:
                    # Single word is too long, break it
                    lines.extend(self._break_long_word(word, font_name, font_size, max_width, canvas_obj))
                    current_line = ""
        
        # Add the last line if there's content
        if current_line:
            lines.append(current_line)
        
        return lines
    
    def _break_long_word(self, word: str, font_name: str, font_size: int, max_width: float, canvas_obj) -> list:
        """Break a single long word that doesn't fit"""
        lines = []
        current_part = ""
        
        for char in word:
            test_part = current_part + char
            test_width = canvas_obj.stringWidth(test_part, font_name, font_size)
            
            if test_width <= max_width:
                current_part = test_part
            else:
                if current_part:
                    lines.append(current_part)
                current_part = char
        
        if current_part:
            lines.append(current_part)
        
        return lines
    
    def _setup_japanese_styles(self):
        """Setup Japanese styles by downloading Noto Sans JP if needed"""
        styles = getSampleStyleSheet()
        
        # Always try to download and register Japanese font
        font_registered = self._download_and_register_japanese_font()
        
        if font_registered:
            # Update all styles to use Japanese font
            font_name = 'NotoSansJP'
            
            # Update existing styles
            styles['Normal'].fontName = font_name
            styles['Title'].fontName = font_name
            styles['Heading1'].fontName = font_name
            styles['Heading2'].fontName = font_name
            styles['Heading3'].fontName = font_name
            
            # Create custom Japanese-optimized styles
            styles.add(ParagraphStyle(
                name='JapaneseNormal',
                parent=styles['Normal'],
                fontName=font_name,
                fontSize=12,
                leading=16
            ))
            
            styles.add(ParagraphStyle(
                name='JapaneseTitle',
                parent=styles['Title'],
                fontName=font_name,
                fontSize=18,
                leading=22
            ))
            
            print(f"Japanese font '{font_name}' registered and applied to styles")
        else:
            print("Failed to register Japanese font, using default fonts")
        
        return styles
    
    def _needs_japanese_font(self) -> bool:
        """Check if Japanese font is needed by testing if default font can handle hiragana"""
        try:
            # Try to create a simple paragraph with Japanese text
            test_text = "こんにちは"  # Simple hiragana
            styles = getSampleStyleSheet()
            Paragraph(test_text, styles['Normal'])
            return True  # Always try to get Japanese font for better support
        except:
            return True
    
    def _download_and_register_japanese_font(self) -> bool:
        """Download Noto Sans JP from Google Fonts and register it"""
        # First check if font is already registered
        try:
            from reportlab.lib.fonts import tt2ps
            if 'NotoSansJP' in pdfmetrics._fonts:
                print("Japanese font already registered")
                return True
        except:
            pass
        
        # Try to download and register TTF version
        return self._download_ttf_japanese_font()
    
    def _download_ttf_japanese_font(self) -> bool:
        """Download TTF version of Japanese font with robust error handling"""
        try:
            # Use only TRUE TTF fonts (not OTF) that reportlab can handle
            font_urls = [
                # Known working TTF Japanese font from Adobe
                "https://github.com/adobe-fonts/source-han-sans/raw/release/Variable/TTF/Subset/SourceHanSansJP-VF.ttf",
                # Alternative: Working Noto Sans JP TTF
                "https://github.com/googlefonts/noto-cjk/raw/main/Sans/Variable/TTF/NotoSansCJKjp-VF.ttf"
            ]
            
            import tempfile
            import os
            
            for font_url in font_urls:
                try:
                    print(f"Attempting to download font from: {font_url}")
                    response = requests.get(font_url, timeout=20, headers={'User-Agent': 'Mozilla/5.0'})
                    
                    if response.status_code == 200:
                        temp_dir = tempfile.gettempdir()
                        font_path = os.path.join(temp_dir, 'NotoSansJP-Regular.ttf')
                        
                        with open(font_path, 'wb') as f:
                            f.write(response.content)
                        
                        print(f"Font downloaded successfully: {len(response.content)} bytes")
                        
                        # Register the font with reportlab
                        try:
                            pdfmetrics.registerFont(TTFont('NotoSansJP', font_path))
                            print("Font registered with reportlab")
                            
                            # Test font registration by checking available fonts
                            registered_fonts = pdfmetrics.getRegisteredFontNames()
                            if 'NotoSansJP' in registered_fonts:
                                print("✓ Font registration confirmed")
                                
                                # Test drawing Japanese text with canvas
                                test_buffer = io.BytesIO()
                                test_canvas = canvas.Canvas(test_buffer)
                                try:
                                    test_canvas.setFont('NotoSansJP', 12)
                                    test_canvas.drawString(100, 100, "こんにちは")
                                    test_canvas.save()
                                    print("✓ Canvas font test successful")
                                    return True
                                except Exception as canvas_e:
                                    print(f"✗ Canvas font test failed: {canvas_e}")
                                    return False
                            else:
                                print("✗ Font not found in registered fonts list")
                                return False
                                
                        except Exception as reg_e:
                            print(f"Font registration error: {reg_e}")
                            continue
                    else:
                        print(f"Font download failed: HTTP {response.status_code}")
                        
                except Exception as e:
                    print(f"Font download error from {font_url}: {e}")
                    continue
            
            print("All font download attempts failed")
            return False
            
        except Exception as e:
            print(f"Font download general error: {e}")
            return False
    
    def _safe_japanese_text(self, text: str) -> str:
        """Return text as-is since we now download Japanese fonts"""
        return str(text) if text else ""
    
    def _parse_pdf_structure(self, story, data: dict, styles):
        """Parse dictionary structure for PDF generation"""
        # Add title if present
        title_keys = ['title', 'タイトル', 'name', 'subject']
        for key in title_keys:
            if key in data and data[key]:
                safe_title = self._safe_japanese_text(str(data[key]))
                story.append(Paragraph(safe_title, styles['Title']))
                story.append(Spacer(1, 12))
                break
        
        # Add metadata if present
        meta_keys = ['date', 'author', 'created']
        meta_info = []
        for key in meta_keys:
            if key in data and data[key]:
                meta_info.append(f"{key.title()}: {data[key]}")
        
        if meta_info:
            safe_meta = self._safe_japanese_text(" | ".join(meta_info))
            story.append(Paragraph(safe_meta, styles['Normal']))
            story.append(Spacer(1, 12))
        
        # Process content
        processed = set(title_keys + meta_keys)
        
        for key, value in data.items():
            if key in processed:
                continue
                
            if key == 'content':
                if isinstance(value, list):
                    for item in value:
                        self._add_pdf_text_content(story, item, styles)
                else:
                    self._add_pdf_text_content(story, value, styles)
            elif key == 'sections':
                if isinstance(value, list):
                    for section in value:
                        if isinstance(section, dict):
                            # Handle section dictionary
                            section_title = section.get('title') or section.get('heading', '')
                            if section_title:
                                safe_title = self._safe_japanese_text(str(section_title))
                                story.append(Paragraph(safe_title, styles['Heading1']))
                            
                            section_content = section.get('content') or section.get('body', '')
                            if section_content:
                                self._add_pdf_text_content(story, section_content, styles)
                            story.append(Spacer(1, 12))
                        else:
                            self._add_pdf_text_content(story, section, styles)
            elif value:
                # Other fields
                if isinstance(value, (list, dict)):
                    safe_key = self._safe_japanese_text(f"{str(key).title()}:")
                    story.append(Paragraph(safe_key, styles['Heading2']))
                    if isinstance(value, list):
                        for item in value:
                            self._add_pdf_text_content(story, item, styles)
                    else:
                        safe_value = self._safe_japanese_text(str(value))
                        story.append(Paragraph(safe_value, styles['Normal']))
                    story.append(Spacer(1, 12))
                else:
                    safe_text = self._safe_japanese_text(f"{str(key).title()}: {value}")
                    story.append(Paragraph(safe_text, styles['Normal']))
                    story.append(Spacer(1, 6))
    
    def _add_pdf_text_content(self, story, content, styles):
        """Add text content to PDF story with proper formatting"""
        if not content:
            return
            
        content = str(content)
        
        # Handle markdown-style headings
        if content.startswith('# '):
            safe_text = self._safe_japanese_text(content[2:])
            story.append(Paragraph(safe_text, styles['Heading1']))
        elif content.startswith('## '):
            safe_text = self._safe_japanese_text(content[3:])
            story.append(Paragraph(safe_text, styles['Heading2']))
        elif content.startswith('### '):
            safe_text = self._safe_japanese_text(content[4:])
            story.append(Paragraph(safe_text, styles['Heading3']))
        elif content.startswith('---'):
            # Horizontal rule - add some space
            story.append(Spacer(1, 12))
        elif content.startswith('|') and '|' in content[1:]:
            # Simple table handling - convert to formatted text
            safe_content = self._safe_japanese_text(content)
            try:
                story.append(Paragraph(f"<font name='Courier'>{safe_content}</font>", styles['Normal']))
            except:
                # Fallback if Courier is not available
                story.append(Paragraph(safe_content, styles['Normal']))
        else:
            # Regular content - handle line breaks
            lines = content.replace('\\n', '\n').split('\n')
            for line in lines:
                line = line.strip()
                if line:
                    # Handle bullet points
                    if line.startswith('- '):
                        safe_text = self._safe_japanese_text(f"• {line[2:]}")
                        story.append(Paragraph(safe_text, styles['Normal']))
                    else:
                        safe_text = self._safe_japanese_text(line)
                        story.append(Paragraph(safe_text, styles['Normal']))
                        
        story.append(Spacer(1, 6))

    def generate_xlsx(self, data: Union[List[Dict], Dict, List[List]], **kwargs) -> bytes:
        """Generate XLSX content with flexible data structure support"""
        if not PANDAS_AVAILABLE:
            raise ImportError("pandas and openpyxl are required for XLSX generation. Install with: pip install pandas openpyxl")
        
        buffer = io.BytesIO()
        
        with pd.ExcelWriter(buffer, engine='openpyxl') as writer:
            if isinstance(data, dict):
                # Handle multiple sheets or complex structure
                sheets_written = False
                
                for key, value in data.items():
                    # Skip metadata keys
                    if key.startswith('_'):
                        continue
                        
                    sheet_name = str(key)[:31]  # Excel sheet name limit
                    
                    try:
                        if isinstance(value, list) and value:
                            if isinstance(value[0], list):
                                # List of lists - treat as raw data with first row as header
                                df = pd.DataFrame(value[1:], columns=value[0] if value else [])
                            elif isinstance(value[0], dict):
                                # List of dictionaries
                                df = pd.DataFrame(value)
                            else:
                                # Simple list - single column
                                df = pd.DataFrame({sheet_name: value})
                        elif isinstance(value, dict):
                            # Look for table-like data in nested dict
                            table_data = self._extract_table_from_dict(value)
                            if table_data:
                                # Found table data
                                if len(table_data) > 1:
                                    df = pd.DataFrame(table_data[1:], columns=table_data[0])
                                else:
                                    df = pd.DataFrame(table_data)
                            else:
                                # Convert dict to key-value pairs
                                df = pd.DataFrame(list(value.items()), columns=['項目', '内容'])
                        else:
                            # Single value
                            df = pd.DataFrame({sheet_name: [value]})
                        
                        df.to_excel(writer, sheet_name=sheet_name, index=False)
                        sheets_written = True
                        
                    except Exception:
                        # If conversion fails, create simple sheet with the data
                        simple_df = pd.DataFrame({sheet_name: [str(value)]})
                        simple_df.to_excel(writer, sheet_name=sheet_name, index=False)
                        sheets_written = True
                
                # If no sheets were written, create a default one
                if not sheets_written:
                    default_df = pd.DataFrame({'Data': ['No valid data found']})
                    default_df.to_excel(writer, sheet_name='Sheet1', index=False)
                    
            elif isinstance(data, list):
                # Handle list data
                if data and isinstance(data[0], list):
                    # List of lists
                    if len(data) > 1:
                        df = pd.DataFrame(data[1:], columns=data[0])
                    else:
                        df = pd.DataFrame(data)
                elif data and isinstance(data[0], dict):
                    # List of dictionaries
                    df = pd.DataFrame(data)
                else:
                    # Simple list
                    df = pd.DataFrame({'Values': data})
                
                df.to_excel(writer, sheet_name='Sheet1', index=False)
            else:
                # Single value or other type
                df = pd.DataFrame({'Data': [str(data)]})
                df.to_excel(writer, sheet_name='Sheet1', index=False)
        
        buffer.seek(0)
        return buffer.read()

    def _extract_table_from_dict(self, data: dict) -> Optional[List[List]]:
        """Extract table-like data from nested dictionary"""
        table_keys = ['table', 'テーブル', 'data', 'データ', 'rows', '行']
        
        for key in table_keys:
            if key in data:
                value = data[key]
                if isinstance(value, list) and value:
                    # Check if it's a proper table (list of lists)
                    if isinstance(value[0], list):
                        return value
        
        # Look for any list of lists in the dict values
        for value in data.values():
            if isinstance(value, list) and value and isinstance(value[0], list):
                return value
                
        return None

    def generate_zip(self, files: Dict[str, Any], **kwargs) -> bytes:
        """Generate ZIP content"""
        buffer = io.BytesIO()
        
        with zipfile.ZipFile(buffer, 'w', zipfile.ZIP_DEFLATED) as zf:
            for filename, content in files.items():
                if isinstance(content, str):
                    zf.writestr(filename, content.encode('utf-8'))
                elif isinstance(content, bytes):
                    zf.writestr(filename, content)
                else:
                    # Try to generate as structured data
                    file_ext = filename.split('.')[-1].lower() if '.' in filename else 'txt'
                    try:
                        file_content = self.generate_content(file_ext, content)
                        if file_content:
                            zf.writestr(filename, file_content)
                    except:
                        # Fallback to string representation
                        zf.writestr(filename, str(content).encode('utf-8'))
        
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
        return content.encode('utf-8')

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
        return content.encode('utf-8')

    def generate_sql(self, data: Union[str, Dict], **kwargs) -> bytes:
        """Generate SQL content"""
        if isinstance(data, str):
            content = data
        else:
            content = f"""-- Generated SQL file
-- Created: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}

{data.get('content', '')}
"""
        return content.encode('utf-8')


def _upload_file(file_content: bytes, filename: str, file_type: str, file_size: int) -> str:
    """Upload file using multiple services with fallback"""
    
    import urllib.parse
    safe_filename = urllib.parse.quote(filename, safe='.-_')
    
    # Try multiple services in order
    services = [
        {
            "name": "transfer.sh",
            "url": f"https://transfer.sh/{safe_filename}",
            "method": "put",
            "retention": "サービス設定に依存"
        },
        {
            "name": "0x0.st", 
            "url": "https://0x0.st",
            "method": "post",
            "retention": "30日～1年（ファイルサイズ依存）"
        },
        {
            "name": "file.io",
            "url": "https://file.io",
            "method": "post", 
            "retention": "14日（1回ダウンロードで削除）"
        }
    ]
    
    errors = []  # Initialize errors list
    
    for service in services:
        try:
            # Add User-Agent header for all requests
            headers = {"User-Agent": "curl/7.68.0"}
            
            if service["method"] == "put":
                # transfer.sh style
                response = requests.put(service["url"], data=file_content, headers=headers, timeout=15)
            else:
                # file.io and 0x0.st style (multipart form)
                files = {"file": (filename, file_content)}
                data = {}
                
                # Add secret parameter for 0x0.st
                if service["name"] == "0x0.st":
                    data["secret"] = ""
                
                response = requests.post(service["url"], files=files, data=data, headers=headers, timeout=15)
            
            if response.status_code == 200:
                download_url = ""
                delete_token = ""
                
                if service["name"] == "file.io":
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
### 🗑️ ファイル削除

0x0.stでファイルを削除するには、以下のコマンドを実行してください：

```bash
curl -F "token={delete_token}" -F "delete=" {download_url}
```

⚠️ **注意**: このコマンドを実行するとファイルが即座に削除されます
"""
                    
                    return f"""## ✅ ファイル生成・アップロード完了

**📄 ファイル名:** `{filename}`  
**💾 サイズ:** {file_size} bytes ({file_size/1024:.1f} KB)  
**🕒 作成日時:** {datetime.now().strftime('%Y年%m月%d日 %H:%M')}  
**🌐 サービス:** {service["name"]}

### 📥 ダウンロード

**[⬇️ {filename} をダウンロード]({download_url})**

### 📋 直接リンク
```
{download_url}
```
{delete_section}
💡 **保持期間**: {service["retention"]}  
🔗 **共有**: このリンクを他の人と共有できます
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
    error_details = "\n".join(errors) if errors else "詳細不明"
    
    return f"""## ❌ 全サービスでアップロード失敗

**ファイル:** `{filename}` ({file_size} bytes)

### デバッグ情報
```
{error_details}
```

試行したサービス: transfer.sh, 0x0.st, file.io

💡 **解決策**: エラー詳細を確認してネットワーク設定をチェックしてください
"""




class Tools:
    """Public API for Open WebUI - file upload approach"""
    
    class Valves(BaseModel):
        """Configuration settings for the Universal File Generator"""
        pass
    
    def __init__(self):
        self.generator = FileGenerator()
        self.valves = self.Valves()

    def generate_file(
        self,
        file_type: str = Field(..., description="File type: csv, json, xml, txt, html, md, docx, pdf, xlsx, zip, js, py, sql"),
        data: Any = Field(..., description="Data to convert to file format"),
        filename: Optional[str] = Field(None, description="Custom filename (optional)"),
        indent: Optional[int] = Field(2, description="JSON indentation (for JSON files)"),
        root_name: Optional[str] = Field("root", description="Root element name (for XML files)"),
        __request__: object = None,
        __user__: dict = {}
    ) -> str:
        """
        Generate a file of specified type from provided data
        Uploads all files to Open WebUI file system
        
        :param file_type: Type of file to generate
        :param data: Data to convert (format varies by file type)
        :param filename: Optional custom filename
        :param indent: JSON indentation level (default: 2)
        :param root_name: XML root element name (default: 'root')
        :return: Markdown with download information
        """
        
        try:
            # Validate file type
            supported_types = ['csv', 'json', 'xml', 'txt', 'html', 'md', 'docx', 'pdf', 'xlsx', 'zip', 'js', 'py', 'sql']
            if file_type not in supported_types:
                return f"❌ Unsupported file type: {file_type}. Supported: {', '.join(supported_types)}"

            # Generate filename if not provided or is Field object
            if not filename or hasattr(filename, 'default'):
                timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
                filename = f"generated_{timestamp}.{file_type}"
            else:
                # Ensure filename is string and has correct extension
                filename = str(filename)
                if not filename.endswith(f'.{file_type}'):
                    filename += f'.{file_type}'

            # Prepare kwargs for file generators
            kwargs = {
                'indent': indent,
                'root_name': root_name
            }

            # Generate file content
            try:
                file_content = self.generator.generate_content(file_type, data, **kwargs)
                
                if file_content is None:
                    return f"❌ Failed to generate {file_type} content"
                    
                if len(file_content) == 0:
                    return f"❌ Generated empty {file_type} content"
                    
            except Exception as e:
                return f"❌ Content generation error: {str(e)}"

            # Unified file processing - always use upload approach
            file_size = len(file_content)
            return _upload_file(file_content, filename, file_type, file_size)

        except Exception as e:
            return f"❌ Unexpected error generating {file_type} file: {str(e)}"

    def list_supported_formats(
        self,
        __request__: object = None,
        __user__: dict = {}
    ) -> str:
        """
        List all supported file formats and their requirements
        
        :return: List of supported formats with availability status
        """
        
        formats_info = {
            'csv': '✅ Always available',
            'json': '✅ Always available', 
            'xml': '✅ Always available',
            'txt': '✅ Always available',
            'html': '✅ Always available',
            'md': '✅ Always available',
            'js': '✅ Always available',
            'py': '✅ Always available', 
            'sql': '✅ Always available',
            'zip': '✅ Always available',
            'docx': '✅ Available' if DOCX_AVAILABLE else '❌ Requires: pip install python-docx',
            'pdf': '✅ Available' if PDF_AVAILABLE else '❌ Requires: pip install reportlab',
            'xlsx': '✅ Available' if PANDAS_AVAILABLE else '❌ Requires: pip install pandas openpyxl'
        }
        
        result = "📋 **Universal File Generator v3.0 - Supported Formats:**\n\n"
        for fmt, status in formats_info.items():
            result += f"• **{fmt.upper()}**: {status}\n"
        
        result += f"\n💡 **Usage example:**\n"
        result += f"```\n"
        result += f"generate_file(\n"
        result += f"  file_type='csv',\n"
        result += f"  data=[{{'name': 'Alice', 'age': 25}}, {{'name': 'Bob', 'age': 30}}],\n"
        result += f"  filename='users.csv'\n"
        result += f")\n"
        result += f"```\n\n"
        result += f"🎯 **New in v4.0:** Smart data parsing & external upload!\n"
        result += f"🚀 **All files:** Multi-service upload with delete tokens\n"
        result += f"🧠 **AI-friendly:** Flexible parsing for any data structure\n"
        result += f"📝 **Open WebUI optimized:** Clean Markdown output"
        
        return result