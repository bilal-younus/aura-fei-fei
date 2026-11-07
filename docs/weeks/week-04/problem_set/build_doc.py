"""Convert the canonical mission-04-doc.md into its submission PDF."""
from pathlib import Path
import re
from xml.sax.saxutils import escape
from reportlab.platypus import SimpleDocTemplate,Paragraph,Spacer,Table,TableStyle,PageBreak,Image
from reportlab.lib.styles import getSampleStyleSheet,ParagraphStyle
from reportlab.lib import colors
ROOT=Path(__file__).resolve().parent
styles=getSampleStyleSheet()
styles.add(ParagraphStyle(name='BodyM',fontName='Helvetica',fontSize=9.3,leading=12.5,spaceAfter=7))
styles.add(ParagraphStyle(name='CellM',fontName='Helvetica',fontSize=8,leading=10.4))
styles.add(ParagraphStyle(name='CodeM',fontName='Courier',fontSize=8.2,leading=11,spaceAfter=5))
styles['Heading1'].textColor=colors.HexColor('#153B50');styles['Heading1'].fontSize=17
styles['Heading2'].fontSize=12;styles['Heading2'].textColor=colors.HexColor('#153B50')
story=[]
def inline(s):
 s=escape(s)
 s=re.sub(r'\[([^\]]+)\]\((https?://[^)]+)\)',r'<link href="\2" color="#126A8D">\1</link>',s)
 s=re.sub(r'\*\*(.+?)\*\*',r'<b>\1</b>',s)
 s=re.sub(r'`([^`]+)`',r'<font name="Courier">\1</font>',s)
 return s

def table(rows):
 n=len(rows[0]);widths={2:[244,244],3:[142,158,188],4:[65,135,135,153]}.get(n,[488/n]*n)
 data=[[Paragraph(inline(c),styles['CellM']) for c in row] for row in rows]
 t=Table(data,colWidths=widths,repeatRows=1,hAlign='LEFT')
 t.setStyle(TableStyle([('BACKGROUND',(0,0),(-1,0),colors.HexColor('#DFECF1')),('VALIGN',(0,0),(-1,-1),'TOP'),('GRID',(0,0),(-1,-1),.4,colors.HexColor('#BCCDD5')),('LEFTPADDING',(0,0),(-1,-1),6),('RIGHTPADDING',(0,0),(-1,-1),6),('TOPPADDING',(0,0),(-1,-1),6),('BOTTOMPADDING',(0,0),(-1,-1),6)]))
 story.extend([t,Spacer(1,8)])

lines=(ROOT/'mission-04-doc.md').read_text().splitlines();i=0;paragraph=[];code=False

def flush():
 if paragraph:story.append(Paragraph(inline(' '.join(paragraph)),styles['BodyM']));paragraph.clear()

while i<len(lines):
 line=lines[i].strip();i+=1
 if line.startswith('```'):flush();code=not code;continue
 if code:
  story.append(Paragraph(escape(line),styles['CodeM']));continue
 if not line:flush();continue
 if line=='<!-- pagebreak -->':flush();story.append(PageBreak());continue
 if line.startswith('|'):
  flush();rows=[]
  while True:
   row=[c.strip() for c in line.strip('|').split('|')]
   if not all(re.fullmatch(r':?-+:?',c) for c in row):rows.append(row)
   if i>=len(lines) or not lines[i].strip().startswith('|'):break
   line=lines[i].strip();i+=1
  table(rows);continue
 m=re.match(r'!\[([^\]]*)\]\(([^)]+)\)',line)
 if m:
  flush();path=ROOT/m.group(2)
  from PIL import Image as PILImage
  with PILImage.open(path) as image:w,h=image.size
  width=435;height=width*h/w
  story.append(Image(str(path),width=width,height=height));story.append(Spacer(1,5));continue
 if line.startswith('# '):flush();story.append(Paragraph(inline(line[2:]),styles['Heading1']));continue
 if line.startswith('## '):flush();story.append(Paragraph(inline(line[3:]),styles['Heading2']));continue
 if line.startswith('- '):flush();story.append(Paragraph('&#8226; '+inline(line[2:]),styles['BodyM']));continue
 paragraph.append(line)
flush()
def footer(c,d):
 c.setFont('Helvetica',8);c.setFillColor(colors.HexColor('#526873'))
 c.drawString(53,26,'SQUADRON FEI-FEI | MISSION 04');c.drawRightString(542,26,str(d.page))
SimpleDocTemplate(str(ROOT/'mission-04-doc.pdf'),pagesize=(595,842),leftMargin=53,rightMargin=54,topMargin=38,bottomMargin=45).build(story,onFirstPage=footer,onLaterPages=footer)
print('Built mission-04-doc.pdf from mission-04-doc.md')
